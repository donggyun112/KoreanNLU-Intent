import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import stanza

class IntentClassifier(nn.Module):
	def __init__(self, bert_model_name="skt/kobert-base-v1", num_classes=5):
		super(IntentClassifier, self).__init__()
		self.bert = BertModel.from_pretrained(bert_model_name)
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
		
	def forward(self, input_ids, attention_mask):
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		pooled_output = outputs.pooler_output
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		return logits

class IntentClassificationSystem:
	def __init__(self, model_dir="intent_models"):
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			self.device = torch.device("mps")
		else:
			self.device = torch.device("cpu")
			
		self.model_dir = model_dir
		self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
		self.model = IntentClassifier(num_classes=5).to(self.device)
		self.nlp = stanza.Pipeline('ko')  # Pipeline을 인스턴스 변수로
		
		self.intent_map = {
			'play.video': 0,
			'search.video': 1,
			'resume.video': 2,
			'set.channel.selected': 3,
			'undefined': 4
		}
		self.id_to_intent = {v: k for k, v in self.intent_map.items()}

	def load_model_file(self, model_path):
		try:
			checkpoint = torch.load(model_path, map_location=self.device)
			
			# 시나리오 1: checkpoint가 state_dict 자체인 경우
			if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
				self.model.load_state_dict(checkpoint)
				
			# 시나리오 2: checkpoint가 model_state_dict를 포함하는 경우
			elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
				self.model.load_state_dict(checkpoint['model_state_dict'])
				
			else:
				raise ValueError("Unknown checkpoint format")
				
			print(f"[모델 로드 완료] {model_path}")
			return True
			
		except Exception as e:
			print(f"[모델 로드 에러] {str(e)}")
			return False

	def predict_with_probs(self, text, threshold=0.8):
		self.model.eval()
		with torch.no_grad():
			doc = self.nlp(text)
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
			logits = outputs
			probs = F.softmax(logits, dim=1)
			max_prob, predicted = torch.max(probs, 1)
			
			# threshold 적용
			if max_prob.item() < threshold:
				predicted = torch.tensor([self.intent_map['undefined']]).to(self.device)
				
		return logits.cpu().numpy(), probs.cpu().numpy(), self.id_to_intent[predicted.item()]

	def predict(self, text):
		_, _, intent = self.predict_with_probs(text)
		return intent

	def open_input_txt_file(filepath):
		result = []
		with open(filepath , 'r', encoding='utf-8') as f:
			texts = f.readlines()
			result = [txt.strip() for txt in texts]
		return result

def main():
	system = IntentClassificationSystem()
	model_path = 'intent_models/intent_model_best_model_epoch15_acc99.25.pt'
	
	load_success = system.load_model_file(model_path)
	if not load_success:
		return
	
	#   sample_texts = open_input_txt_file('input.txt')
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

	print("\n[샘플 입력에 대한 예측 결과]")
	for txt in sample_texts:
		logits, probs, intent = system.predict_with_probs(txt, threshold=0.8)
		print(f"\n입력: '{txt}'")
		print(f"예측 의도: {intent}")
		print(f"로짓(Logits): {logits}")
		print(f"확률(Probabilities): {probs}")

if __name__ == "__main__":
	main()