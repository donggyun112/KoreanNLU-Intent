import os
import json
from datetime import datetime

import torch
from torch import nn
import pandas as pd
import numpy as np

from transformers import BertModel
from torch.utils.data import DataLoader, Dataset

# scikit-learn으로 train/val/test 분할
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# KoBERT 토크나이저 임포트
from kobert_tokenizer import KoBERTTokenizer

# Stanza를 사용한 한국어 구문 분석
import stanza

# 한국어 모델 다운로드 및 파이프라인 초기화
stanza.download('ko')  # 최초 실행 시 다운로드
nlp = stanza.Pipeline('ko')


class IntentClassifier(nn.Module):
    def __init__(self, bert_model_name="skt/kobert-base-v1", num_classes=5):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # BERT의 [CLS] 토큰 벡터
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = [int(label) if not pd.isna(label) else 0 for label in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 구문 분석 수행
        doc = nlp(text)
        dependencies = []
        for sentence in doc.sentences:
            for word in sentence.words:
                dependencies.append(f"{word.head}-{word.deprel}")
        
        # 의존 관계를 문자열로 변환
        dep_str = ' '.join(dependencies)
        
        # 텍스트와 의존 관계를 결합
        combined_text = f"{text} [SEP] {dep_str}"
        
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),     # (1, max_len) -> (max_len,)
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def balance_classes(df, target_ratio=0.2, majority_class='undefined'):
    """
    언더샘플링으로 majority_class의 비율을 target_ratio로 맞춤.
    예: undefined 데이터를 전체의 20% 비율로 제한
    """
    class_counts = df['intent'].value_counts()
    print("\n[1차 언더샘플링] 클래스별 샘플 수:")
    print(class_counts)
    
    majority = df[df['intent'] == majority_class]
    minority = df[df['intent'] != majority_class]
    
    target_majority = int(len(df) * target_ratio)
    
    if len(majority) > target_majority:
        majority_downsampled = majority.sample(target_majority, random_state=42)
        print(f"\n'{majority_class}' 클래스 샘플을 {len(majority_downsampled)}개로 줄였습니다.")
    else:
        majority_downsampled = majority
        print(f"\n'{majority_class}' 클래스 수가 이미 {target_majority} 이하입니다.")
    
    balanced_df = pd.concat([minority, majority_downsampled], ignore_index=True)
    print("\n[1차 언더샘플링] 완료 후 클래스별 샘플 수:")
    print(balanced_df['intent'].value_counts())
    
    return balanced_df


def balance_classes_additional(df, target_ratio=0.3, majority_class='play.video'):
    """
    추가로 play.video 클래스도 언더샘플링해서 비율을 target_ratio로 맞춤.
    예: play.video를 전체의 30% 비율로 제한
    """
    class_counts = df['intent'].value_counts()
    print("\n[2차 언더샘플링] 클래스별 샘플 수:")
    print(class_counts)
    
    majority = df[df['intent'] == majority_class]
    minority = df[df['intent'] != majority_class]
    
    target_majority = int(len(df) * target_ratio)
    
    if len(majority) > target_majority:
        majority_downsampled = majority.sample(target_majority, random_state=42)
        print(f"\n'{majority_class}' 클래스 샘플을 {len(majority_downsampled)}개로 줄였습니다.")
    else:
        majority_downsampled = majority
        print(f"\n'{majority_class}' 클래스 수가 이미 {target_majority} 이하입니다.")
    
    balanced_df = pd.concat([minority, majority_downsampled], ignore_index=True)
    print("\n[2차 언더샘플링] 완료 후 클래스별 샘플 수:")
    print(balanced_df['intent'].value_counts())
    
    return balanced_df


#####################
# (1) 템플릿 기반 대량 생성 함수들 추가
#####################

def generate_search_sentences(keywords, patterns):
    """
    search.video 클래스를 위한 템플릿 기반 문장 대량 생성
    keywords: ["축구 경기", "야구 하이라이트", "최신 예능", ...]
    patterns: ["{kw} 어디서 볼 수 있나요?", "{kw} 검색해봐", ...]
    """
    result = []
    for kw in keywords:
        for pat in patterns:
            new_text = pat.format(kw=kw)
            result.append({'text': new_text, 'intent': 'search.video'})
    return result

def generate_resume_sentences(titles, patterns):
    """
    resume.video 클래스를 위한 템플릿 기반 문장 대량 생성
    titles: ["슬기로운 의사생활", "응답하라 1988", ...]
    patterns: ["{title} 멈춘 지점부터 이어보기", ...]
    """
    result = []
    for t in titles:
        for pat in patterns:
            new_text = pat.format(title=t)
            result.append({'text': new_text, 'intent': 'resume.video'})
    return result

def generate_channel_sentences(channels, patterns):
    """
    set.channel.selected 클래스를 위한 템플릿 기반 문장 대량 생성
    channels: ["KBS", "MBC", "SBS", ...]
    patterns: ["{ch} 틀어줘", "{ch} 생방송 보여줘", ...]
    """
    result = []
    for ch in channels:
        for pat in patterns:
            new_text = pat.format(ch=ch)
            result.append({'text': new_text, 'intent': 'set.channel.selected'})
    return result


def augment_data(df):
    """데이터 증강 + 템플릿 대량생성 + 언더샘플링으로 클래스 불균형 해결"""
    
    # --------------------
    # (A) 기존 증강 로직
    # --------------------
    
    augmentation_patterns = {
        'play.video': {
            '틀어줘': ['보여줘', '재생해줘', '플레이해줘', '켜줘', '틀어주세요', '보여주세요', 
                      '재생해주세요', '플레이해주세요', '켜주세요'],
            '틀어': ['재생해', '보여줘', '플레이해', '켜줘', '시작해'],
            '보기': ['시청', '감상', '재생', '플레이'],
            '보고싶어': ['보고 싶어요', '시청하고싶어요', '보고 싶은데', '시청하고 싶어'],
            '볼래': ['볼 수 있을까', '보면 좋겠어', '보고 싶을까'],
            '부탁해': ['부탁드려요', '부탁드립니다', '부탁합니다', '부탁해요'],
            '틀어줄래': ['틀어주시겠어요', '틀어주실 수 있나요', '틀어주실래요'],
            '몰아보기': ['몰아서 보기', '정주행', '몰아서 보여줘', '정주행 틀어줘'],
            '다시보기': ['다시 보고 싶어', '리플레이 해줘', '한번 더 보기'],
            '나와' : ['나와줘 ', '나와주세요', '나와라', '나와서 보여줘'],
        },
        'search.video': {
            '찾아줘': ['검색해줘', '찾아봐', '찾아볼래', '검색해볼래', '찾아주세요', 
                      '검색해주세요', '찾아달라고', '검색해달라고'],
            '알려줘': ['알아봐줘', '알려주세요', '알려주실래요', '알아봐주세요'],
            '뭐 있어': ['뭐가 있어', '어떤게 있어', '뭐가 있는지', '무엇이 있나요'],
            '알고싶어': ['알고 싶은데', '알아보고 싶어', '찾아보고 싶어']
        },
        'resume.video': {
            '이어보기': ['이어서 보기', '멈춘 곳부터', '계속 보기', '다시 보던 거', 
                         '이어서 재생', '계속해서 보기', '멈춘 부분부터', '계속 재생'],
            '계속보기': ['계속해서 보기', '계속 재생', '이어서 보기']
        }
    }
    
    additional_augmentation = {
        'search.video': {
            '유튜브 검색': ['유튜브 찾아', '유튜브 찾아봐', '유튜브 검색해', '유튜브 검색해줘', '유튜브에서 검색'],
            '검색 요청': ['유튜브에서 찾아줘', '유튜브에서 검색해줘', '유튜브에서 검색 부탁해']
        },
        'resume.video': {
            '이어보기': ['이어서 보기', '멈춘 곳부터', '계속 보기', '다시 보던 거', 
                         '이어서 재생', '계속해서 보기', '멈춘 부분부터', '계속 재생'],
            '계속보기': ['계속해서 보기', '계속 재생', '이어서 보기']
        }
    }
    
    # --------------------
    # (B) 기존 증강 로직 실행
    # --------------------
    
    augmented_data = []
    for idx, row in df.iterrows():
        text = row['text']
        intent = row['intent']
        
        if intent == 'undefined':
            continue
        if intent in augmentation_patterns:
            patterns = augmentation_patterns[intent]
            lower_text = text.lower()
            for old, new_list in patterns.items():
                if old in lower_text:
                    for new_w in new_list:
                        new_text = lower_text.replace(old, new_w)
                        if new_text != lower_text:
                            augmented_data.append({'text': new_text, 'intent': intent})
    
    # 추가 패턴 로직
    for intent, pat_dict in additional_augmentation.items():
        for old_pattern, new_list in pat_dict.items():
            for idx, row in df.iterrows():
                if row['intent'] != intent:
                    continue
                lower_text = row['text'].lower()
                if old_pattern in lower_text:
                    for new_w in new_list:
                        new_text = lower_text.replace(old_pattern, new_w)
                        if new_text != lower_text:
                            augmented_data.append({'text': new_text, 'intent': intent})
    
    # --------------------
    # (C) 템플릿 기반 대량 생성
    # --------------------
    
    # 예시 키워드 리스트 (자유롭게 확장 가능)
    search_keywords = [
        "축구 경기", "야구 하이라이트", "새로운 예능", "최신 영화",
        "유명 팝송 뮤직비디오", "인기 드라마", "과학 다큐", "환경 문제 영상",
        "버스킹 공연", "라이브 콘서트", "요리 레시피", "헬스 운동 영상",
        "아이돌 춤 연습", "자동차 리뷰", "테크놀로지 최신 동향",
        "여행 브이로그", "게임 스트리밍", "뉴스 속보", "교육 강의",
        "라이프스타일 블로그", "DIY 프로젝트", "패션 트렌드", "뷰티 튜토리얼"
    ]
    search_patterns = [
        "{kw} 어디서 볼 수 있나요?",
        "{kw} 검색해봐",
        "{kw} 찾아봐",
        "{kw} 알아보고 싶어",
        "혹시 {kw} 영상 있나?",
        "{kw} 관련 리뷰 좀 볼래",
        "어디서 {kw}을/를 시청할 수 있어?",
        "{kw} 클립을 검색해줘",
        "온라인에서 {kw} 찾아봐",
        "{kw} 관련 정보를 찾아줘",
        "{kw} 비디오를 찾아줘",
        "인터넷에서 {kw} 검색해줘"
    ]
    
    # resume.video용 타이틀 & 템플릿
    resume_titles = [
        "슬기로운 의사생활", "응답하라 1988", "아워블루", "더글로리", "무빙",
        "멈춘 다큐", "휴먼 다큐 사랑", "여행 스케치 영상", "미니시리즈 드라마",
        "인간실격", "마법사와 마녀", "로맨틱 코미디", "역사 드라마", "범죄 스릴러",
        "판타지 모험", "리얼리티 쇼", "시즌1 재생", "시즌2 이어보기", "시즌3 다시보기",
        "TV 시리즈", "온라인 드라마", "스릴러 영화", "액션 시리즈", "판타지 영화",
        "드라마 시리즈", "코미디 프로그램", "스포츠 리그", "버추얼 리얼리티"
    ]
    resume_patterns = [
        "{title} 멈춘 지점부터 이어보기",
        "중단한 {title} 다시 볼래",
        "일시정지했던 {title} 계속 시청",
        "{title} 이어서 보자",
        "중단된 {title} 다시 재생",
        "{title} 이어서 플레이해줘",
        "이전에 보던 {title} 이어서 재생",
        "마지막으로 보던 {title} 이어서 보기",
        "{title} 계속 재생",
        "잠시 멈춘 {title} 이어서 시청",
        "{title}을/를 이어서 보여줘",
        "중단했던 {title} 이어서 재생해줘",
        "멈춘 {title}을/를 이어서 시청할래",
        "{title} 다시 이어서 볼 수 있어?",
        "이어보기 위해 {title} 재생"
    ]
    
    # set.channel.selected 용
    channel_templates = [
        "{ch} 방송",
        "{ch} 틀어줘",
        "{ch} 방송 틀어줘",
        "{ch} 채널",
        "{ch} 방송 보여줘",
        "{ch} 라이브",
        "{ch} 실시간",
        "{ch} 생방송",
        "{ch} 채널 틀어줘",
        "{ch} 채널로 전환해줘",
        "{ch} 채널 시청",
        "{ch} 채널 켜줘",
        "{ch} 채널 재생",
        "{ch} 방송으로 변경",
        "{ch} 채널을 켜줘",
        "지금 {ch} 채널 틀어줘",
        "바로 {ch} 방송 보여줘",
        "즉시 {ch} 채널 재생",
        "{ch} 라이브 방송 켜줘",
        "{ch} 실시간 방송 재생"
    ]
    channels_list = [
        "KBS", "MBC", "SBS", "JTBC", "tvN", "채널A", "MBN", 
        "TV조선", "YTN", "연합뉴스TV", "KBS1", "KBS2", 
        "MBC Every1", "OCN", "EBS", "Discovery", "Nickelodeon", 
        "HBO", "Netflix", "Amazon Prime", "Hulu", "BBC", "CNN", "ESPN",
        "Disney Channel", "TBS", "TVN Movies", "Apple TV", "Fox", "Sky"
    ]
    
    # (C1) 대량 생성된 문장
    big_search_data = generate_search_sentences(search_keywords, search_patterns)
    big_resume_data = generate_resume_sentences(resume_titles, resume_patterns)
    big_channel_data = generate_channel_sentences(channels_list, channel_templates)
    
    # augmented_data에 합침
    augmented_data.extend(big_search_data)
    augmented_data.extend(big_resume_data)
    augmented_data.extend(big_channel_data)
    
    # --------------------
    # (D) DataFrame 변환 & 병합
    # --------------------
    aug_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    # --------------------
    # (E) 언더샘플링 (undefined -> 20%, play.video -> 30%)
    # --------------------
    
    # undefined 20%
    combined_df = balance_classes(combined_df, target_ratio=0.2, majority_class='undefined')
    
    # play.video 30%
    combined_df = balance_classes_additional(combined_df, target_ratio=0.3, majority_class='play.video')
    
    # --------------------
    # (F) 소수 클래스 오버샘플링
    # --------------------
    
    def oversample_minority_classes(df, target_counts):
        """
        소수 클래스에 대해 오버샘플링을 수행합니다.
        target_counts: dict {intent: target_count}
        """
        df_list = [df]
        for intent, target in target_counts.items():
            current_count = df['intent'].value_counts().get(intent, 0)
            if current_count < target:
                needed = target - current_count
                # 현재 클래스 데이터
                class_data = df[df['intent'] == intent]
                if len(class_data) == 0:
                    print(f"'{intent}' 클래스에 대한 데이터가 없습니다. 오버샘플링을 건너뜁니다.")
                    continue
                # 고유한 증강을 위해 특수 문자 추가
                for _ in range(needed):
                    sample = class_data.sample(n=1, replace=True, random_state=None).iloc[0]
                    # 간단한 노이즈 추가 (예: 끝에 특수 문자 추가)
                    noisy_text = sample['text'] + "!"
                    df_list.append(pd.DataFrame([{'text': noisy_text, 'intent': intent}]))
                print(f"'{intent}' 클래스에 대해 {needed}개의 샘플을 오버샘플링했습니다.")
        oversampled_df = pd.concat(df_list, ignore_index=True)
        return oversampled_df
    
    # 소수 클래스 목표 샘플 수 설정 (예: search.video: 1000, resume.video: 500, set.channel.selected: 500)
    target_counts = {
        'search.video': 1000,
        'resume.video': 500,
        'set.channel.selected': 500
    }
    
    combined_df = oversample_minority_classes(combined_df, target_counts)
    
    # --------------------
    # (G) 중복 제거 (오버샘플링 이후)
    # --------------------
    
    # 소수 클래스는 중복 제거에서 제외
    majority_intents = ['play.video', 'undefined']
    majority_df = combined_df[combined_df['intent'].isin(majority_intents)]
    minority_df = combined_df[~combined_df['intent'].isin(majority_intents)]
    
    # majority_df에서 중복 제거
    majority_df['text_lower'] = majority_df['text'].str.lower()
    majority_df = majority_df.drop_duplicates(subset=['text_lower'])
    majority_df = majority_df.drop(columns=['text_lower'])
    
    # 소수 클래스는 그대로 유지
    final_df = pd.concat([majority_df, minority_df], ignore_index=True)
    
    print("\n[최종] 증강 & 클래스 비율 조정 후 데이터 분포:")
    print(final_df['intent'].value_counts())
    
    return final_df


class IntentClassificationSystem:
    def __init__(self, model_dir="intent_models"):
        # 장치 설정: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model_dir = model_dir
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.model = IntentClassifier(num_classes=5).to(self.device)
        self.training_history = []
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.intent_map = {
            'play.video': 0,
            'search.video': 1,
            'resume.video': 2,
            'set.channel.selected': 3,
            'undefined': 4
        }
        self.id_to_intent = {v: k for k, v in self.intent_map.items()}
        
        self.criterion = None
        self.optimizer = None

    def load_training_data(self, csv_path):
        """CSV 파일에서 학습 데이터 로드 + 증강 + 언더샘플링 → train/val/test 분할"""
        try:
            df = pd.read_csv(csv_path)
            print("\n[원본 데이터 미리보기]")
            print(df.head())
            
            required_cols = ['text', 'intent']
            if not all(c in df.columns for c in required_cols):
                raise ValueError("CSV에 'text', 'intent' 컬럼이 필요합니다.")
            
            # 전처리
            df['text'] = df['text'].astype(str).str.strip()
            df = df[df['text'] != '']
            
            # undefined 처리
            df['intent'] = df['intent'].fillna('undefined').str.strip()
            valid_intents = set(self.intent_map.keys())
            df.loc[~df['intent'].isin(valid_intents), 'intent'] = 'undefined'
            
            # 데이터 증강 및 클래스 불균형 해결
            print("\n[데이터 증강 + 언더샘플링 시작]")
            augmented_df = augment_data(df)
            print("\n[데이터 증강 + 언더샘플링 완료]")
            
            augmented_df['label'] = augmented_df['intent'].map(self.intent_map)
            
            # train/val/test 분할
            train_df, temp_df = train_test_split(
                augmented_df,
                test_size=0.3,
                random_state=42,
                stratify=augmented_df['label']
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=42,
                stratify=temp_df['label']
            )
            
            train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
            val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
            test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()
            
            print(f"\n[분할 결과]")
            print(f"  - Train: {len(train_texts)}개")
            print(f"  - Val:   {len(val_texts)}개")
            print(f"  - Test:  {len(test_texts)}개")
            
            return (train_texts, train_labels,
                    val_texts, val_labels,
                    test_texts, test_labels)
            
        except Exception as e:
            print(f"[에러] {e}")
            import traceback
            traceback.print_exc()
            return [], [], [], [], [], []
    
    def save_model(self, version=None):
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.model_dir, f"intent_model_{version}.pt")
        history_path = os.path.join(self.model_dir, f"training_history_{version}.json")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'version': version,
        }, model_path)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
            
        print(f"[모델 저장] {model_path}")
    
    def load_model(self, version="latest"):
        if version == "latest":
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("intent_model_")]
            if not model_files:
                print("[모델 로드 실패] 저장된 모델이 없습니다.")
                return False
            version = sorted(model_files)[-1].split("intent_model_")[-1].split(".")[0]
        
        model_path = os.path.join(self.model_dir, f"intent_model_{version}.pt")
        history_path = os.path.join(self.model_dir, f"training_history_{version}.json")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.training_history = json.load(f)
                    
            print(f"[모델 로드 완료] {model_path}")
            return True
        
        except Exception as e:
            print(f"[모델 로드 에러] {e}")
            return False

    def _build_dataloader(self, texts, labels, batch_size=16, shuffle=True):
        dataset = IntentDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def train(self, 
              train_texts, train_labels, 
              val_texts=None, val_labels=None,
              epochs=10, batch_size=16):
        
        train_loader = self._build_dataloader(train_texts, train_labels, batch_size, shuffle=True)
        
        # 클래스별 가중치 계산
        label_counts = pd.Series(train_labels).value_counts()
        weights = torch.tensor([
            1.0 / label_counts[i] if i in label_counts else 1.0 for i in range(5)
        ], device=self.device, dtype=torch.float32)
        weights = weights / weights.sum() * len(weights)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        best_val_acc = 0
        best_epoch = 0
        best_model_state = None
        
        self.model.train()
        epoch_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = total_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            val_loss, val_acc = 0, 0
            if val_texts and val_labels:
                val_loss, val_acc = self.evaluate(val_texts, val_labels, batch_size)
            
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            epoch_history.append({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = self.model.state_dict().copy()
        
        # 최고 성능 모델 불러오기
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n[최고 성능 모델] Epoch {best_epoch}, Val Acc: {best_val_acc:.2f}%")
        
        self.training_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epochs': epochs,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'num_train_samples': len(train_texts)
        })
        
        return epoch_history

    def evaluate(self, texts, labels, batch_size=16):
        eval_loader = DataLoader(IntentDataset(texts, labels, self.tokenizer),
                                 batch_size=batch_size, shuffle=False)
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label_batch = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, label_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()
        
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0
        avg_acc = 100.0 * correct / total if total > 0 else 0
        
        return avg_loss, avg_acc

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            # 구문 분석 수행
            doc = nlp(text)
            dependencies = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    dependencies.append(f"{word.head}-{word.deprel}")
            
            dep_str = ' '.join(dependencies)
            combined_text = f"{text} [SEP] {dep_str}"
            
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
            _, predicted = torch.max(outputs, 1)
        return self.id_to_intent[predicted.item()]
    
    def plot_confusion_matrix(self, texts, labels):
        self.model.eval()
        preds = []
        trues = []
        
        for txt, lbl in zip(texts, labels):
            pred_intent = self.predict(txt)
            preds.append(pred_intent)
            trues.append(self.id_to_intent[lbl])
        
        cm = confusion_matrix(trues, preds, labels=list(self.intent_map.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.intent_map.keys()))
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.show()
        
        # Classification report
        print("\n[Classification Report]")
        print(classification_report(trues, preds, labels=list(self.intent_map.keys())))
    
    def save_predictions(self, texts, predictions, output_path='predictions.csv'):
        df = pd.DataFrame({
            'text': texts,
            'predicted_intent': predictions
        })
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[예측 결과 저장] {output_path}")


def main():
    system = IntentClassificationSystem()
    
    # 기존 모델 로드
    load_success = system.load_model(version="20250102_additional_finetune")
    if not load_success:
        print("[주의] 기존 모델을 로드하지 못했습니다. 새로운 모델로 학습을 진행합니다.")
    
    # CSV 파일로부터 데이터 로드 → 증강 + 언더샘플링 → 분할
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        system.load_training_data('example_training_data2.csv')
    
    if train_texts and train_labels:
        print("\n[모델 학습 시작]")
        system.train(train_texts, train_labels, val_texts, val_labels, epochs=5, batch_size=16)
        
        # 모델 저장 (새 버전명 지정)
        system.save_model(version="20250103_additional_finetune")
        
        # 테스트셋 평가
        print("\n[테스트셋 평가]")
        test_loss, test_acc = system.evaluate(test_texts, test_labels)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 예시 문장 예측
        sample_texts = [
            "넷플릭스 영화 틀어줘",
            "유튜브 검색해줘",
            "멈춘 곳부터 다시 보여줘",
            "정주행 틀어줘",
            "KBS 채널 틀어줘",
            "EBS 생방송 보여줘",
            "응답하라 1988 이어서 보기",
            "오션블루 다큐 멈춘 부분부터 재생",
            "MBC 실시간 방송 켜줘",
            "유튜브에서 환경 다큐 검색 부탁해",
            ""
        ]
        preds = []
        for txt in sample_texts:
            intent = system.predict(txt)
            preds.append(intent)
            print(f"입력: '{txt}' -> 예측 의도: {intent}")
        
        # 예측 결과 저장
        system.save_predictions(sample_texts, preds, 'predictions.csv')
        
        # 혼동 행렬 시각화
        print("\n[혼동 행렬 시각화]")
        system.plot_confusion_matrix(test_texts, test_labels)


if __name__ == "__main__":
    main()
