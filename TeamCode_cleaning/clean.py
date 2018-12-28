# -*- coding: utf-8 -*-

def clean_data(df):
    import re
    print("### Data is Cleaning ###")
    def preprocessor(text):
        text = re.sub('<[^>]*>','',text)
        emoticons = re.findall('(?::|;=)(?:-)?(?:\)|\(|D|P)',text)
        text = (re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-',''))
        return text
    df['review'] = df['review'].apply(preprocessor)
    print("### Data is Cleaned ###")
    return df