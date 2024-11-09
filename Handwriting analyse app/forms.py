from django import forms
from .models import Image

class ImageForm(forms.ModelForm):

    class Meta:
        model = Image
        fields = '__all__' 
        labels = {'image' : ""}
    # def __init__(self, *args, **kwargs):
    #     super(ImageForm, self).__init__(*args, **kwargs)
    #     self.fields['image'].widget.attrs={
    #         'id': 'file-input'}
