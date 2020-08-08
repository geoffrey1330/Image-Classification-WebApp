from django import forms

CATEGORY_CHOICE = (
    ('Face Mask', 'Face Mask'),
    ('Smile' ,'Smile')
)


class ClassifierForm(forms.Form):
    image = forms.ImageField(required=True)
    category = forms.ChoiceField(choices=CATEGORY_CHOICE)