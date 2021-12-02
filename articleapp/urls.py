from django.urls import path
from django.views.generic import TemplateView

from articleapp.views import ArticleDetailView

app_name = 'articleapp'

urlpatterns = [
    path('detail/', ArticleDetailView.as_view(), name='detail')
]