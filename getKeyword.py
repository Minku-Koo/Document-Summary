'''
21.03.25
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re, json
from eunjeon import Mecab

from krwordrank.word import KRWordRank
from krwordrank.hangle import normalize
from krwordrank.sentence import summarize_with_sentences

from libs.makeWordCloud import make_wordcloud
from textrankr import TextRank

'''
기능 생각
> 키워드 3~8개 설정 (mecab 여부)
> 워드 클라우드 생성 (mecab 적용 > 명사만)
> 감성 분석 모델 적용(감성 단위 5단계 구분)
> 문서 요약 - 문장 추출 (문장 3~6개)(mecab 여부)
> 문서 요약 - 추상 추출 (딥러닝)

Mecab()
* N~ : 체언
* V~ : 동사
M~ : 부사
IC : 감탄사
J~ : 조사
E~ : 어말 어미
X~ : 접두/접미사
S~ : 부호
* SL/SN : 외국어/숫자
'''

dirpath = "./train/"

# json read
def readJson(dirpath, filename):
    with open(dirpath + filename + '.jsonl', 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)

    return [ json.loads(js) for js in json_list ]

# mecab nlp preprocessing
# only 동사, 명사, 영어
# input : 한 문장 -> output : 정제된 한 문장 String to String
def mecab_process(data):
    mcb = Mecab()
    data_processed = mcb.pos(data)
    result = " "
    for index, word in enumerate(data_processed):
        morph = word[1] #
        
        if morph[0] in ["N", "V"] or  morph=="SL":
            if index!=0 and data_processed[index-1][1][0] == "N":
                result += " "
            result +=  word[0] + " "
    result = re.sub(' {2}', " ", result)
    return result.strip()
    # return " ".join(result)

def build_tfidf(data_list):
    result = []
    vectorizer = TfidfVectorizer()
    sp_matrix = vectorizer.fit_transform(data_list)
    
    word2id = defaultdict(lambda : 0)
    for idx, feature in enumerate(vectorizer.get_feature_names()):
        word2id[feature] = idx
    for i, sent in enumerate(data_list):
        result.append( [ (token, sp_matrix[i, word2id[token]]) for token in sent.split() ] )
        
    return result

d = """
[스포티비뉴스=서재원 기자] 레알 마드리드가 해리 케인(토트넘 홋스퍼) 영입전에 뛰어 들었다.

글로벌 축구 전문 매체 '포포투'는 24일(한국시간) "레알 마드리드가 케인 영입전을 위해 맨체스터 유나이티드와 경쟁한다"라고 보도했다. 

손흥민의 단짝 케인의 미래가 안갯속에 빠졌다. 토트넘의 다음 시즌 유럽축구연맹(UEFA) 챔피언스리그 진출이 불투명한 상황에서 케인이 떠날 수도 있다는 불안감이 점점 커지고 있다.

케인은 토트넘을 통해 잉글랜드 프리미어리그(PL)를 넘어 세계 최고의 공격수로 성장했다. 하지만, 채우지 못한 부분이 있었다. 최고 전성기를 누비고 있음에도 우승과 인연이 없었기 때문이다. 우승은 물론, 챔피언스리그에도 뛸 수 없는 상황이라면 이적은 당연히 고려해야 할 부분이다.

케인도 우승에 대한 열망을 숨기지 않았다. 그는 지난해 초 현지 언론과 인터뷰에서 우승을 위해 토트넘을 떠날 수 있다는 뜻을 내비쳤다. 신종 코로나바이러스 감염증(코로나19) 팬데믹에 따른 토트넘의 재정적 상황 때문에 케인을 팔 수 밖에 없다는 전망도 있다.

케인이 떠난다면, 레알 마드리드가 유력 행선지가 될 수 있다. 포포투는 "레알 마드리드는 케인을 1순위 이적시장 타깃으로 삼았다. 그의 계약을 위해 맨유와 경쟁할 것"이라며, 스페인 '카데나세르'의 보도를 인용해 "레알 마드리드가 카림 벤제마에 대한 의존도를 낮추기 위해 그를 지켜보고 있다"라고 주장했다.

포포투는 이어 "얼링 홀란드(보루시아 도르트문트)와 킬리안 음바페(파리 생제르망) 모두 레알 마드리드와 연결돼 있었지만, 그들은 케인에게 관심을 돌린 것으로 보인다"라고 설명을 덧붙였다.
"""

d1= """
 다비드 데 헤아(31)의 입지가 예전만 못하다. 맨체스터 유나이티드를 떠날 수 있다는 소식이 심심찮게 들린다.
영국 매체 '데일리 메일'은 24일(이하 한국 시간) "맨유가 토트넘 주전 골키퍼 위고 요리스를 데 헤아의 대체 선수로 고려하고 있다. 토트넘이 새로운 골키퍼를 영입한다면, 이적 시장에서 요리스를 노릴 것"이라고 밝혔다.

최근 토트넘은 요리스와 이별을 준비하는 모양새다. 토트넘이 유럽축구연맹(UEFA) 유로파리그에서 탈락하자 요리스는 인터뷰에서 좋지 않은 라커룸 분위기를 공개적으로 전하며 팀에 불만을 드러냈다.

요리스는 토트넘과 2022년 여름이면 계약이 끝난다. 1986년생으로 요리스의 나이가 적지 않은 점을 고려하면, 토트넘으로선 이번 여름 이적 시장이 요리스를 제 값을 받고 팔 절호의 기회다.

맨유는 이런 토트넘과 요리스의 이상기류를 면밀히 지켜보고 있다. 2011년부터 팀의 주전 골키퍼로 활약한 데 헤아의 이적을 놓고 고심 중이기 때문이다.

이미 맨유는 딘 핸더슨이라는 뛰어난 골키퍼가 백업으로 있다. 올 시즌 데 헤아가 자리를 비울 때도 큰 공백은 느껴지지 않았다.

1990년생인 데 헤아의 나이도 어느덧 30살을 넘겼다. 기량은 점점 내려가는데 주급은 팀 내 가장 많은 35만 파운드(약 5억 5천만 원)를 받고 있다.

맨유는 데 헤아를 내보내면 핸더슨을 주전으로 올리거나 요리스를 영입해 주전 수문장을 바꾸려는 계획을 품고 있다. 남은 시즌 데 헤아, 핸더슨, 요리스의 경기력에 따라 이 계획은 언제든 바뀔 수 있다.

토트넘은 새로운 골키퍼를 모색 중이다. 영국 매체 '메트로'는 24일 "토트넘은 잉글랜드 국가대표에 뽑힌 핸더슨, 샘 존스톤, 닉 포프에 관심이 있다. 핸더슨은 맨유에서 주전이 아니면 떠날 수 있고 존스톤, 포프는 더 큰 팀으로 이적 가능성이 충분한 상태"라고 알렸다.
"""

# 키워드 추출
# min_count : 최소 출현 횟수 / max_len : 단어 최대 길이 / top : 키워드 상위 n 개
def extractKeyword( data, min_count =3, max_len = 10, top = 20 ):
    
    #data = data.replace("\n", "").split(".")
    data = [normalize(text, english=True, number=True) for text in data]
    # 결과를 관찰하고 필요에 따라 이 부분에 mecab 추가
    data =  [ mecab_process(d) for d in data ]
    
    make_wordcloud(" ".join(data), "word clouds test")

    wordrank_extractor = KRWordRank(
        min_count = min_count, # 단어의 최소 출현 빈도수 (그래프 생성 시)
        max_length = max_len, # 단어의 최대 길이
        verbose = True
        )

    beta = 0.85    # PageRank의 decaying factor beta
    max_iter = 10

    # 키워드 추출을 위해 원본으로 키워드 추출
    keywords, rank, graph = wordrank_extractor.extract(data, beta, max_iter)

    for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:top]:
        print('%8s:\t%.4f' % (word, r))
        pass
    
    return sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:top]


# 문서 요약
# stopwords : 불용어 리스트 / min_count : 최소 출현 횟수 / max_len : 단어 최대 길이 
# keyword_extract : 문서 요약에 필요한 키워드 사전 개수 / sentences : 문장 추출 개수
def document_summary(org_data, stopwords={}, min_count =3, max_len = 10, keyword_extract = 80, sentences = 4 ):
    #org_data = data.replace("\n", "").split(".")
    
    data =  [ mecab_process(d) for d in org_data ]

    beta = 0.85    # PageRank의 decaying factor beta
    max_iter = 10

    wordrank_extractors = KRWordRank(
        min_count = min_count, # 단어의 최소 출현 빈도수 (그래프 생성 시)
        max_length = max_len, # 단어의 최대 길이
        verbose = True
        )
    # 문서 요약 위해 정제한 것으로 문서 요약 / 결과를 지켜보고 org data로 변경 가능
    keywords, rank, graph = wordrank_extractors.extract(data, beta, max_iter)


    penalty = lambda x:0 if (15 <= len(x) <= 80) else 1
    stopwords = {'너무', '정말', '진짜'} # 불용어 추후에 추가

    keywords, sents = summarize_with_sentences(
        data,
        penalty=penalty,
        stopwords = stopwords,
        diversity=0.5, # 높을 수록 다양한 문장
        num_keywords=keyword_extract, # 키워드 추출 개수
        num_keysents=sentences, # 요약된 문장 개수
        verbose=False
    )
    print("doc summary keyword:", keywords)

    for s in sents : print("processed data>", s)

    # 정제된 데이터를 원본 데이터와 매칭
    abstracts_data = []
    for line in sents:
        for d in org_data:
            for indx, l in enumerate( line.split(" ") ):
                if l not in d: break
                    
                else:
                    if indx == len(line.split(" "))-1 :
                        abstracts_data.append( d )

    for s in abstracts_data : print("original data>", s)
    
    return abstracts_data

from typing import List
class MecabTokenizer:
    mcb = Mecab()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.mcb.morphs(text)
        return tokens

def text_summary(data):
    data =  [ mecab_process(d) for d in data ]
    data = " ".join(data)

    textRank = TextRank(MecabTokenizer())
    k = 1
    summarized = textRank.summarize(data, k)
    print("summarized:",summarized) 


    summaries = textRank.summarize(data, k, verbose=False)
    for summary in summaries:
        print("summary:",summary)
        pass




if __name__ == "__main__":
    data = readJson(dirpath, "train")[0]["article_original"]
    #data = d1.replace("\n", "").split(".")
    print( " ".join(data) )
    data = d.replace("\n", "").split(".")
    # 키워드 추출
    keywords = extractKeyword( data,  min_count =3, max_len = 10, top = 20)


    # -------------------
    # -------------------

    # 문서 요약
    document_summary(data, min_count =3, max_len = 10, keyword_extract = 80, sentences = 4 )

    text_summary(data)
    