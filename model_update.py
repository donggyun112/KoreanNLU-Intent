import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import torch.nn.functional as F
from datetime import datetime
import stanza


# -----------------------------------
# 1) Stanza 초기화
# -----------------------------------
stanza.download('ko')
nlp = stanza.Pipeline('ko')


# -----------------------------------
# 2) 모델 정의 (Bert + Classifier)
# -----------------------------------
class SimpleIntentClassifier(nn.Module):
    def __init__(self, bert_model_name="skt/kobert-base-v1", num_classes=5):
        super(SimpleIntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
       
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)


# -----------------------------------
# 3) Dataset 정의
#   (구문 분석 → POS/DEP/HEAD → 문자열로 합치기)
# -----------------------------------
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.nlp = nlp  # 전역 파이프라인 사용
   
    def __len__(self):
        return len(self.texts)
   
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
       
        # 구문 분석
        doc = self.nlp(text)
        features = []
        for sentence in doc.sentences:
            for word in sentence.words:
                features.append(f"POS={word.upos}")
                features.append(f"DEP={word.deprel}")
                features.append(f"HEAD={word.head}")
       
        combined_text = f"{text} [SEP] {' '.join(features)}"
       
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
       
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# -----------------------------------
# 4) Trainer 구현
# -----------------------------------
class AdditionalPatternTrainer:
    def __init__(self):
        # GPU/CPU 할당
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
       
        # 모델 디렉토리 생성
        self.model_dir = "intent_models"
        os.makedirs(self.model_dir, exist_ok=True)
       
        # 토크나이저, 모델 준비
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        self.model = SimpleIntentClassifier().to(self.device)
       
        # Intent 라벨 맵
        self.intent_map = {
            'play.video': 0,
            'search.video': 1,
            'resume.video': 2,
            'set.channel.selected': 3,
            'undefined': 4
        }


    def load_pretrained(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            print(f"[모델 로드 완료] {model_path}")
            return True
        except FileNotFoundError:
            print(f"[경고] 모델 파일을 찾을 수 없습니다: {model_path}")
            return False
        except Exception as e:
            print(f"[경고] 모델 로드 중 에러 발생: {str(e)}")
            return False


    def freeze_bert_except_last_layers(self, unfrozen_layer_count=1):
        """
        BERT 마지막 N개 레이어를 제외하고 모두 Freeze.
        (unfrozen_layer_count=2 → BERT 마지막 두 레이어만 학습)
        """
        # 먼저 전부 Freeze
        for param in self.model.bert.parameters():
            param.requires_grad = False
       
        # 각 레이어 모듈 이름: bert.encoder.layer.X
        total_layers = len(self.model.bert.encoder.layer)
        unfrozen_start = total_layers - unfrozen_layer_count
       
        for idx in range(total_layers):
            if idx >= unfrozen_start:
                for param in self.model.bert.encoder.layer[idx].parameters():
                    param.requires_grad = True
       
        # Pooler까지 조정하고 싶으면, 아래 주석 해제
        # for param in self.model.bert.pooler.parameters():
        #     param.requires_grad = True


    def train_additional_pattern(self, train_texts, train_labels, epochs=5, batch_size=16, freeze_mode='partial', patience=3, min_loss_change=0.0001):
        """
        패턴을 추가로 학습하는 메서드
        
        Args:
            train_texts (list): 학습할 텍스트 목록
            train_labels (list): 학습할 레이블 목록
            epochs (int): 전체 학습 에폭 수
            batch_size (int): 배치 크기
            freeze_mode (str): BERT 레이어 동결 모드
                - 'partial': BERT 일부 레이어만 파인튜닝
                - 'all': 전부 동결 (classifier만 학습)
                - 'none': BERT 전부 학습
            patience (int): Early stopping patience
            min_loss_change (float): Loss 개선 판단을 위한 최소 변화량
            
        Returns:
            tuple: (best_epoch, best_accuracy)
        """
        # Freeze 설정
        if freeze_mode == 'all':
            for param in self.model.bert.parameters():
                param.requires_grad = False
        elif freeze_mode == 'partial':
            self.freeze_bert_except_last_layers(unfrozen_layer_count=2)
        elif freeze_mode == 'none':
            for param in self.model.bert.parameters():
                param.requires_grad = True
        
        # 분류기 파라미터는 항상 학습
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # DataLoader 생성
        train_dataset = SimpleDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 옵티마이저와 손실 함수 설정
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=2e-6)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping 관련 변수들
        best_accuracy = 0
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        prev_loss = float('inf')
        
        # 학습 루프
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 에폭별 성능 계산
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
            
            # Loss 개선 정도 계산
            loss_improvement = prev_loss - avg_loss
            prev_loss = avg_loss
            
            # 모델 성능 평가 (accuracy와 loss 모두 고려)
            is_accuracy_better = accuracy >= best_accuracy
            is_loss_better = avg_loss < best_loss
            is_loss_improved = loss_improvement > min_loss_change
            
            # 모델 저장 조건: accuracy가 같거나 높고, loss가 개선되었거나 최고 기록을 갱신
            if is_accuracy_better and (is_loss_better or is_loss_improved):
                best_accuracy = accuracy
                best_loss = avg_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                best_epoch = epoch + 1
                print(f"New best model found! Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
                print(f"Loss improvement: {loss_improvement:.6f}")
            else:
                patience_counter += 1
                if not is_loss_improved:
                    print(f"Loss improvement ({loss_improvement:.6f}) below threshold ({min_loss_change})")
                if not is_accuracy_better:
                    print(f"Accuracy ({accuracy:.2f}%) not improved from best ({best_accuracy:.2f}%)")
            
            # Early stopping 체크
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # 최고 성능 모델로 복원
        print(f"\nRestoring best model from epoch {best_epoch} with accuracy {best_accuracy:.2f}%")
        self.model.load_state_dict(best_model_state)
        
        return best_epoch, best_accuracy


    def predict_with_threshold(self, text, threshold=0.5):
        """
        threshold 기반으로 가장 높은 확률이 'threshold' 미만이면 'undefined'로 후처리
        """
        self.model.eval()
        with torch.no_grad():
            # 구문 분석
            doc = nlp(text)
            features = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    features.append(f"POS={word.upos}")
                    features.append(f"DEP={word.deprel}")
                    features.append(f"HEAD={word.head}")
           
            combined_text = f"{text} [SEP] {' '.join(features)}"
            encoding = self.tokenizer(
                combined_text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
           
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
           
            outputs = self.model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=-1)
            max_prob, predicted_idx = torch.max(probs, dim=1)
           
            # threshold 이하이면 강제로 undefined
            if max_prob.item() < threshold:
                return 'undefined', probs[0].cpu().tolist()
            else:
                return list(self.intent_map.keys())[predicted_idx.item()], probs[0].cpu().tolist()


    def save_model(self, version=None):
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"intent_model_{version}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"[모델 저장] {model_path}")


# -----------------------------------
# 5) 예시: "나와라" 패턴 → undefined 데이터 보강
# -----------------------------------
def generate_resume_patterns(titles):
    """
    '다시보기' 패턴을 생성하는 함수
    """
    resume_patterns = [
        # 기본 다시보기 패턴
        "{title} 다시보기",
        "{title} 다시 보기",
        "{title} 다시보기 해줘",
        "{title} 다시 보여줘",
        "{title} 다시 틀어줘",
        # 정중한 표현
        "{title} 다시 보여주세요",
        "{title} 다시 틀어주세요",
        
        # 추가 변형
        "{title} 계속 보기",
        "{title} 계속 보여줘",
        "{title} 계속해서 보기",
        "{title} 다시 재생",
        "{title} 이어재생"
    ]
    
    # 에피소드 관련 접미사
    suffixes = ["", "1화", "2화", "3화", "마지막화", "최신화", "예고편"]
    
    result = []
    
    # resume.video 패턴 생성
    for title in titles:
        # 기본 제목
        for pattern in resume_patterns:
            result.append((pattern.format(title=title), 'resume.video'))
        
        # 접미사가 있는 경우
        for suffix in suffixes:
            if suffix:
                title_with_suffix = f"{title} {suffix}"
                for pattern in resume_patterns:
                    result.append((pattern.format(title=title_with_suffix), 'resume.video'))
    
    return result


# -----------------------------------
# 6) main() 예시
# -----------------------------------
def main():
    trainer = AdditionalPatternTrainer()
    
    # 사전 학습된 모델 로드
    model_path = "intent_models/intent_model_20250103_additional_finetune.pt"
    load_success = trainer.load_pretrained(model_path)
    if not load_success:
        print("사전학습된 모델 없이 새로 학습을 시작합니다.")
    
    # 학습 데이터 준비
    content_titles = [
        # 예능
        "인간극장", "무한도전", "런닝맨", "1박2일", "놀면뭐하니",
        "유퀴즈", "나혼자산다", "전국노래자랑", "복면가왕",
        
        # 드라마
        "더글로리", "무빙", "경찰수업", "이번생도잘부탁", "구경이",
        "슬기로운의사생활", "하늘의인연", "천원짜리변호사",
        
        # 다큐멘터리
        "다큐멘터리", "인간의조건", "역사스페셜", "자연의신비",
        "동물의왕국", "우주탐사", "문명의기록", "과학다큐",
    ]
    
    # 다시보기 패턴 추가
    resume_data = generate_resume_patterns(content_titles)
    
    # 데이터 통합
    train_data = resume_data
    
    # 학습 데이터 준비
    train_texts = [x[0] for x in train_data]
    train_labels = [trainer.intent_map[x[1]] for x in train_data]
    
    # 학습 실행
    best_epoch, best_accuracy = trainer.train_additional_pattern(
        train_texts,
        train_labels,
        epochs=5,
        batch_size=16,
        freeze_mode='partial',  # 'partial' 유지
        patience=2,
        min_loss_change=0.0001
    )
    
    # 최고 성능 모델 저장
    version = f"best_model_epoch{best_epoch}_acc{best_accuracy:.2f}"
    trainer.save_model(version)
    
    # 테스트 케이스
    test_sentences = [
        # resume.video 테스트
        "무한도전 다시보기",
        "더글로리 2화 이어보기",
        "런닝맨 다시 보여줘",
        "1박2일 이어서 보기",
        "슬기로운의사생활 3화 다시보기",
        "구경이 이어재생",
        "인간극장 계속 보기",
        
        # 기존 play.video 테스트
        "무한도전 나와라",
        "더글로리 2화 나와라",
        
        # undefined 테스트
        "금 나와라 뚝딱",
        "복 나와라~"
    ]
    
    print("\n[Threshold 예측 테스트]")
    for sentence in test_sentences:
        pred_intent, probs = trainer.predict_with_threshold(sentence, threshold=0.5)
        print(f"입력: '{sentence}' → 예측: {pred_intent}, 확률 분포: {probs}")


import torch
import json
from collections import defaultdict


def test_model(trainer, test_cases=None):
    """
    다양한 테스트 케이스로 모델을 테스트합니다.
    """
    if test_cases is None:
        test_cases = {
            'play.video': [
                "인간극장 나와라",
                "런닝맨 나와줘",
                "무한도전 1화 나와라",
                "더글로리 예고편 나와주세요",
                "범죄도시 2화 나와줘",
            ],
            'undefined': [
                "금 나와라 뚝딱",
                "요술봉 나와라",
                "복 나와라!",
                "황금 나와라~",
                "제발 도깨비 나와라",
            ],
            'edge_cases': [
                "나와라",  # 최소 입력
                "이상한나라의앨리스나와라",  # 학습되지 않은 콘텐츠
                "금은보화 나와라 뚝딱뚝딱",  # 변형된 패턴
                "도깨비랑 무한도전이랑 나와라",  # 복합 입력
                "나와라 인간극장",  # 역순 입력
            ]
        }
   
    results = defaultdict(list)
    confusion = defaultdict(int)
    total_correct = 0
    total_samples = 0


    print("\n=== 모델 테스트 시작 ===")
   
    # 카테고리별 테스트
    for category, sentences in test_cases.items():
        print(f"\n[{category} 테스트]")
        for sent in sentences:
            pred_intent, probs = trainer.predict_with_threshold(sent, threshold=0.5)
           
            # 정답 레이블 결정 (edge_cases는 정확도 계산에서 제외)
            if category != 'edge_cases':
                is_correct = (
                    (category == pred_intent) or
                    (category == 'undefined' and pred_intent == 'undefined')
                )
                total_correct += int(is_correct)
                total_samples += 1
               
                # 혼동 행렬 업데이트
                confusion[f"{category}->{pred_intent}"] += 1
           
            # 결과 저장
            results[category].append({
                'input': sent,
                'predicted': pred_intent,
                'probabilities': [float(p) for p in probs],
                'max_prob': float(max(probs))
            })
           
            # 결과 출력
            print(f"\n입력: {sent}")
            print(f"예측: {pred_intent}")
            print(f"확률 분포: {[f'{p:.4f}' for p in probs]}")
            if category != 'edge_cases':
                print(f"정답 여부: {'O' if is_correct else 'X'}")
   
    # 전체 정확도 계산
    if total_samples > 0:
        accuracy = (total_correct / total_samples) * 100
        print(f"\n전체 정확도: {accuracy:.2f}% ({total_correct}/{total_samples})")
   
    # 혼동 행렬 출력
    print("\n[혼동 행렬]")
    for k, v in confusion.items():
        print(f"{k}: {v}회")
   
    # 결과를 JSON 파일로 저장
    test_results = {
        'accuracy': accuracy if total_samples > 0 else None,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'confusion_matrix': dict(confusion),
        'detailed_results': dict(results)
    }
   
    save_path = 'test_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print(f"\n테스트 결과가 {save_path}에 저장되었습니다.")
   
    return test_results


# 사용 예시:
def run_model_test():
    trainer = AdditionalPatternTrainer()
   
    # 저장된 모델 불러오기
    model_path = "intent_models/intent_model_best_model_epoch15_acc99.25.pt"
    load_success = trainer.load_pretrained(model_path)
   
    if not load_success:
        print("모델 로드 실패")
        return


   
    # 2. input.txt 파일 읽기
    print("\n=== 추가 데이터 테스트 ===")
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            additional_texts = [line.strip() for line in f.readlines() if line.strip()]
       
        print(f"\n총 {len(additional_texts)}개의 추가 텍스트를 읽었습니다.")
       
        # 결과 저장용 딕셔너리
        additional_results = {
            'play.video': [],
            'search.video': [],
            'resume.video': [],
            'set.channel.selected': [],
            'undefined': []
        }
       
        # 각 텍스트에 대해 예측
        print("\n[추가 데이터 예측 결과]")
        for text in additional_texts:
            pred_intent, probs = trainer.predict_with_threshold(text, threshold=0.5)
            max_prob = max(probs)
           
            additional_results[pred_intent].append({
                'text': text,
                'confidence': max_prob,
                'prob_dist': probs
            })
           
            print(f"\n입력: {text}")
            print(f"예측: {pred_intent}")
            print(f"확률 분포: {[f'{p:.4f}' for p in probs]}")
       
        # 3. 통합 분석 결과 출력
        print("\n=== 통합 분석 결과 ===")
       
        # 추가 데이터 분석
        print("\n[추가 데이터 분석]")
        total = len(additional_texts)
       
        # 클래스별 분포
        print("\n클래스별 분포:")
        for intent, items in additional_results.items():
            count = len(items)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"\n{intent}: {count}개 ({percentage:.1f}%)")
           
            if items:
                avg_conf = sum(item['confidence'] for item in items) / len(items)
                print(f"- 평균 신뢰도: {avg_conf:.3f}")
                # 상위 3개 예시 출력
                print("- 상위 3개 예시:")
                for item in sorted(items, key=lambda x: x['confidence'], reverse=True)[:3]:
                    print(f"  • '{item['text']}' (신뢰도: {item['confidence']:.3f})")
       
        # 신뢰도 분포
        print("\n신뢰도 분포:")
        all_confidences = []
        for items in additional_results.values():
            all_confidences.extend(item['confidence'] for item in items)
       
        high_conf = len([c for c in all_confidences if c > 0.9])
        med_conf = len([c for c in all_confidences if 0.6 <= c <= 0.9])
        low_conf = len([c for c in all_confidences if c < 0.6])
       
        print(f"- 높은 신뢰도 (>0.9): {high_conf}개 ({high_conf/total*100:.1f}%)")
        print(f"- 중간 신뢰도 (0.6-0.9): {med_conf}개 ({med_conf/total*100:.1f}%)")
        print(f"- 낮은 신뢰도 (<0.6): {low_conf}개 ({low_conf/total*100:.1f}%)")
       
    except FileNotFoundError:
        print("input.txt 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"에러 발생: {str(e)}")


if __name__ == "__main__":
    main()