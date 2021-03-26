'''
21.03.27
'''
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# make word cloud
# 워드 클라우드 생성
def make_wordcloud(text, title, stopword = ["이다", "있다"], word_max = 50, w = 1200, h = 1000):
    save_path = "./web/static/"
    stopwords = set(STOPWORDS) 
    for word in stopword: stopwords.add(word) 

    # 이 부분에 필요하면 mecab 돌려서 명사만 추출되도록 하면 좋겠다.
    # 일단 대용으로 한 글자 이상만 걸러냄
    data = " ".join([ word if len(word)>1 else ""  for word in text.split() ])

    wordcloud = WordCloud(font_path= save_path+'font/BMDOHYEON_ttf.ttf', 
                        background_color='white',
                        stopwords= stopwords,
                        max_words=word_max, 
                        max_font_size=200, 
                        height=h, 
                        width=w).generate(data)
    
    plt.imshow(wordcloud, interpolation='lanczos') #이미지의 부드럽기 정도
    plt.axis('off') # x y 축 숫자 제거
    plt.savefig(save_path+ 'wordclouds/wordcloud.png', dpi=400) 
    return 

