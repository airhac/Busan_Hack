from django.forms import ModelForm

from articleapp.models import Article


class ArticleCreationForm(ModelForm):
    class Meta:
        model = Article
        fields = ['content']
        #form에서 user한테 입력 받는 내용은 content밖에 없다
