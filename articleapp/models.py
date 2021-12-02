from django.db import models

# Create your models here.
from django.db import models


class Article(models.Model):
    content = models.TextField(max_length=255,null=False)
    label = models.IntegerField(default=0,null=False)