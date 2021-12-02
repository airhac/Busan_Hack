import os

from django.views.generic.list import MultipleObjectMixin

from articleapp.data import Data
from articleapp.dataclassifier import BERTClassifier
import torch
from articleapp.dataset import BERTDataset

modulePath = os.path.dirname(__file__)
print(modulePath)
filePath = os.path.join(modulePath, 'models\\model.pth')

# Create your views here.
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView, FormMixin

from articleapp.forms import ArticleCreationForm
from articleapp.models import Article
import numpy as np

class ArticleDetailView(CreateView):
    model = Article
    context_object_name = 'target_comment'
    form_class = ArticleCreationForm
    #우리가 볼려는 게시물의 객체
    template_name = 'articleapp/detail.html'
    success_url = reverse_lazy('articleapp:detail')

    def form_valid(self, form):
        temp_article = form.save(commit=False)
        logits = test_sentence(temp_article.content)

        if np.argmax(logits) == 1:
            temp_article.label = 1 #악플
        elif np.argmax(logits) == 0:
            temp_article.label = 0 #선플

        temp_article.save()
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        article_list = Article.objects.all()
        return super().get_context_data(**kwargs, article_list=article_list)


max_len = 128
batch_size = 64

m_data = Data()
bert_model = m_data.bert()
model = BERTClassifier(bert_model,  dr_rate=0.5)  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
device = torch.device('cpu')
model.load_state_dict(torch.load(filePath, map_location=device), strict=False)  # state_dict를 불러 온 후, 모델에 저장
device = torch.device("cuda:0")
model.to(device)


def convert_input_data(sentences):
    m_data = Data()
    tok = m_data.tokenizer()
    # BERT의 토크나이저로 문장을 토큰으로 분리
    data = [sentences, '0']
    dataset_another = [data]

    dataset = m_data.make_dataset(dataset_another, tok, max_len)
    print(dataset)
    data_loader = m_data.make_dataloader(dataset, batch_size)

    return data_loader


def test_sentence(sentences):
    # 문장을 입력 데이터로 변환
    data_loader = convert_input_data(sentences)

    # 평가모드로 변경
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(data_loader):
        # 데이터를 GPU에 넣음
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        print(token_ids)
        print(segment_ids)
        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
    return logits