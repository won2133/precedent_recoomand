#프론트에 보낼 데이터(딕셔너리) 생성
def set_x_dic(x_list, word, total):
    x_dic = {}
    x_dic['x_list'] = x_list #데이터 리스트
    x_dic['word'] = word #검색 단어/카테고리
    x_dic['total'] = total #전체 개수
    return x_dic
    
#판례 불러오기(api 이용)
def pan_api(word, opt): #opt == 0이면 카테고리별 판례, opt == 1이면 검색
    #query(단어)가 포함된 판례의 일련번호, 사건종류명(민사, 형사 등) 등을 반환하는 api
    u1 = ('http://www.law.go.kr/DRF/lawSearch.do?OC=ID&search=2&target=prec&type=XML&display=100&page=')
    #일련번호를 이용하여 판례(사건명, 선고일자, 참조조문, 판시사항, 판결요지, 판례내용 등)을 반환하는 api
    u2 = ('http://www.law.go.kr/DRF/lawService.do?OC=ID&type=XML&target=prec&ID=')
    prece_list = []
    w = parse.quote(word)
    pan_list = ['판시사항', '판결요지', '판례내용'] #검색 결과(미리보기)로 보여줄 내용 후보
    t = 1
    res = '사건명'
    while True:
        url = u1 + str(t) + '&query=' + w
        print(url)
        try:
            xml = REQ.urlopen(url).read()
            soup = BeautifulSoup(xml, "xml")
            nums = soup.select('판례일련번호') #일련번호 리스트
            kinds = soup.select('사건종류명') #사건종류명 리스트
            if len(nums) == 0:
                break
             #nums 이용해 판례 불러옴
            for k in range(len(nums)):
                if kinds[k].text == '민사': #사건 종류가 민사인 것들만 저장
                    try:
                        w = parse.quote(nums[k].text)
                        url = u2 + w
                        xml = REQ.urlopen(url).read()
                        soup = BeautifulSoup(xml, "xml")
                        try:
                                r = soup.find(res) #사건명
                                r = r.text.split(word)
                                new_r = [] #사건명을 단어로 분리한 리스트(검색 단어를 굵게 표시하기 위함)
                                if len(r) == 1:
                                    new_r = r
                                else:
                                        #공백을 단어로 바꾸기
                                    for i in range(len(r)):
                                        if r[i] == '' or r[i] == ' ':
                                            new_r.append(word)
                                            c += len(word)
                                        else:
                                            new_r.append(r[i])
                                            c += len(r[i])
                                            if i != len(r) - 1:
                                                if r[i + 1] != '' and r[i + 1] != ' ':
                                                    new_r.append(word)
                                                    c += len(word)
                        except:
                                pass
                        if opt == 1: #검색일 때
                            pan = '' # 후보들(pan_list)- 판시사항, 판결요지, 판례내용
                            c = 0 #각 후보(pan_list)에 단어가 있으면 1, 없으면 0
                            new_s = [] # 후보들(pan_list) 중 검색 단어가 있는 것을 검색 단어로 분리하여 저장
                            for p in pan_list:
                                if c > 60:
                                    break
                                try:
                                    pan = soup.find(p)
                                    if pan.find(word) != -1: # 검색한 단어가 있으면 검색 단어를 기준으로 분리하여 리스트에 저장
                                        pan = pan.text.split('.') #문장으로 분리
                                        while c < 60: #총 60 글자가 넘어 가지 않도록
                                            l = pan.pop()
                                            if len(l) < len(word):
                                                continue
                                            l = l.split("<")
                                            if l[0].find(word) != -1: #단어가 있다면
                                                s_list = l[0].split(word)
                                                     #공백을 단어로 바꾸기
                                                for i in range(len(s_list)):
                                                    if c > 60:
                                                        break
                                                    if s_list[i] == '':
                                                        new_s.append(word)
                                                        c += len(word)
                                                    elif s_list[i] == ' ':
                                                        new_s.append(word + ' ')
                                                        c += len(word) + 1
                                                    else:
                                                        new_s.append(s_list[i])
                                                        c += len(s_list[i])
                                                        if i != len(s_list) - 1:
                                                            if s_list[i + 1] != '' and s_list[i + 1] != ' ':
                                                                new_s.append(word)
                                                                c += len(word)
                                            if len(pan) == 0:
                                                break
                                except:
                                    pass
                            if c != 0:  #후보들(pan_list) 중 검색 단어가 포함된 게 있으면 prece_list에 추가
                                prece_list.append([new_r, new_s, nums[k].text])
                        else: #카테고리별 판례(opt=0)
                            prece_list.append([0, nums[k].text, r])
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)
        t += 1
    return prece_list

#검색
@bp.route('/api/search')
def search_index():
    prece_list = []
    a_dic = {} #민법 조항 딕셔너리
    p_dic = {} #판례 딕셔너리
    content = request.args.get('query', type=str)
    opt = request.args.get('option', type=int, default=1) #검색 옵션(1: 전체, 2:민법, 3:판례)
    page = request.args.get('page', type=int, default=1)  #페이지
    opt_list = ['통합검색', '민법', '판례']
    if opt != 3:
        #article db에서 민법 전체 불러오기
        db_name = 'article'
        conn = mysql.connector.connect(user=f'{user_name}', password=f'{pass_my}',
                              host=f'{host_my}',
                              database=f'{db_name}')
        cursor = conn.cursor(prepared=True)
        sql_l = ['''SELECT * FROM article_1;''', '''SELECT * FROM article_2;''', '''SELECT * FROM article_3;''', '''SELECT * FROM article_4;''', '''SELECT * FROM article_5;''']
        result = []
        for sql in sql_l:
            with conn.cursor() as cur:
               cur.execute(sql)
               result.extend(cur.fetchall())
        conn.close()
        article_list = [] #전체 리스트(단어가 있는 조항의 a_list로 이루어진 리스트)
        a_list = [] #각 조항의 조항 제목과 내용을 각각 검색 단어로 분할한 리스트
        for i in range(len(result)):
            check = 0 #i번째 조항에 단어가 있으면 1, 없으면 0
            a = list(result[i]) #i번째 조항 리스트- [조항 제목, 내용, 레이블]
            for k in range(len(a)-1): #맨 뒤는 레이블이므로 제외
                if k == 0:
                    #조항 제목에 해당 단어가 있을 때
                    if a[k].find(content) != -1:
                        check = 1
                        s_list = a[k].split(content)
                        new_s = [] #s_list를 정리하여 저장
                          #공백을 단어로 바꾸기
                        for i in range(len(s_list)):
                            if s_list[i] == '':
                                new_s.append(content)
                            elif s_list[i] == ' ':
                                new_s.append(content + ' ')
                            else:
                                new_s.append(s_list[i])
                                if i != len(s_list) - 1:
                                    if s_list[i + 1] != '' and s_list[i + 1] != ' ':
                                        new_s.append(content)
                        a_list.append(new_s)
                    else:
                        a_list.append([a[k]])
                else:
                      #각 문장 별로 확인
                    for s in a[k][1:len(a[k])-1].split("',"):
                        s = s.replace("'", "")
                        if s.find(content) != -1:
                            check = 1
                            s_list = s.split(content)
                            new_s = [] #s_list를 정리하여 저장
                               #공백을 단어로 바꾸기
                            for i in range(len(s_list)):
                                if s_list[i] == '':
                                    new_s.append(content)
                                elif s_list[i] == ' ':
                                    new_s.append(content + ' ')
                                else:
                                    new_s.append(s_list[i])
                                    if i != len(s_list) - 1:
                                        if s_list[i + 1] != '' and s_list[i + 1] != ' ':
                                            new_s.append(content)
                            a_list.append(new_s)
                        else:
                            a_list.append([s, ' '])
             #해당 조항에 단어가 있으면 article_list에 추가
            if check == 1:
                    article_list.append(a_list)
            a_list = []
        l = len(article_list)
        a_dic = set_x_dic(article_list, content, l)
    if opt != 2:
        prece_list = pan_api(content, 1) #[일련번호, 사건명, 내용]으로 이루어진 2차원 리스트(opt 1)
        l = len(prece_list)
        p_dic = set_x_dic(prece_list, content, l)
        num_list = []  #일련번호 리스트
        for p in prece_list:
            num_list.append(p[1])
        session["num_list"] = num_list #session storage에 num_list 저장
    data = {}
    data["p_dic"] = p_dic
    data["a_dic"] = a_dic
    data["opt"] = opt
    data["opt_list"] = opt_list
    return json.dumps(data, ensure_ascii=False)


#카테고리별 민법  
@bp.route('/api/article/<string:c1>/<int:c2>')
def category_article(c1, c2):
    c_dic = {'총칙': 0, '물권': 1, '채권': 2, '친족': 3, '상속': 4} #파일 이름을 지정하기 위한 딕셔너리
    #카테고리에 해당하는 조항 전체 불러오기
    db_name = 'article'
    conn = mysql.connector.connect(user=f'{user_name}', password=f'{pass_my}',
                              host=f'{host_my}',
                              database=f'{db_name}')
    cursor = conn.cursor(prepared=True)
    sql_l = ['''SELECT * FROM article_1 WHERE label = %s;''', '''SELECT * FROM article_2 WHERE label = %s;''', '''SELECT * FROM article_3 WHERE label = %s;''', '''SELECT * FROM article_4 WHERE label = %s;''', '''SELECT * FROM article_5 WHERE label = %s;''']
    result = []
    sql = sql_l[c_dic[c1]] #c1은 상위카테고리(str)
    cursor.execute(sql, [str(c2)]) #c2는 하위 카테고리(int)
    result = cursor.fetchall()
    conn.close()
    a_list = []
    for r in result:
        contents = r[1].split("',")
        a = []
        for c in contents:
            c = c.replace("'", "")
            c = c.replace("[", "")
            c = c.replace("]", "")
            a.append(c)
        a_list.append([r[0], a])
    l = len(a_list) 
    a_dic = set_x_dic(a_list, c2, l)
    data = {}
    data["c1"] = c1
    data["c2"] = c2
    data["dic"] = a_dic
    return json.dumps(data, ensure_ascii=False)

#카테고리별 판례
@bp.route('/api/precedent/<string:c1>/<string:c2>')
def category_pan(c1, c2):
    prece_list = []
    page = request.args.get('page', type=int, default=1)  # 페이지
    prece_list = pan_api(c2, 0) #[일련번호, 사건명, 0]으로 이루어진 2차원 배열(opt=0)
    last_num = 0
    l = len(prece_list)
    p_dic = set_x_dic(prece_list, c2, l)
    num_list = []  #일련번호 리스트
    for p in prece_list:
        num_list.append(p[1])
   session["num_list"] = num_list #session storage에 num_list 저장
    data = {}
    data["c1"] = c1
    data["c2"] = c2
    data["dic"] = p_dic
return json.dumps(data, ensure_ascii=False)


kkma = Kkma()
# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','과','도','를','으로','자','에','와','한','하다',
             '적극', '소극','여부', '되다', '제', '매', '로', '때', '후', '로', '전', '민법', '방법',
             '경우', '상', '따르다', '있다', '않다', '원심', '및', '법', '에서', '또는', '그', '수', '에게',
             '인지', '해당', '에게', '위', '판결', '조', '인', '위', '사례', '사안', '대하', '되어다'
             '효력', '판단', '청구', '소송', '법원', '제기', '인정', '의미', '요건', '받다', '취지',
             '는지', '관하', '다고']
tokenized_data = []
#토큰화 함수
def tokenize_sentence(sentence):
    tokenized_sentence = kkma.pos(sentence)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word[0] in stopwords] # 불용어 제거
    l = []
    for s in stopwords_removed_sentence:
      l.append(s)
    return l

#임베딩 함수
def get_document_vectors(document_list):
    document_embedding_list = []
    loaded_model = FastText.load("/var/www/law/src/pan_20000_fst_kk_2") # 모델 로드
    #각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line[0].split():
            try:
                w = loaded_model.wv.get_vector(word)
                count += 1

                #해당 문서에 있는 모든 단어들의 벡터 값을 더한다.
                if doc2vec is None :
                    doc2vec = w
                else :
                    doc2vec = doc2vec + w
            except:
                pass

        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec/count
            document_embedding_list.append(doc2vec)
        else:
            document_embedding_list.append(None)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list


#kobert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()
#파라미터 설정
max_len = 64
batch_size = 32
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
    def __len__(self):
        return (len(self.labels))
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
        
#모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model_state_dict = torch.load("/var/www/myapp/src/bert_q_7400_epoch10_vs1_dict.pt")
model.load_state_dict(model_state_dict)

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            if np.argmax(logits) == 0:
                test_eval.append("손해배상")
            elif np.argmax(logits) == 1:
                test_eval.append("민사일반")
            elif np.argmax(logits) == 2:
                test_eval.append("물권")
            elif np.argmax(logits) == 3:
                test_eval.append("채권")
            elif np.argmax(logits) == 4:
                test_eval.append("계약")
            elif np.argmax(logits) == 5:
                test_eval.append("친족")
            elif np.argmax(logits) == 6:
                test_eval.append("상속")
        res = ">> 입력하신 문장은 " + test_eval[0] + " 관련 내용입니다."
        print(res) #결과 확인
        return np.argmax(logits)


#판례 추천
@bp.route('/api/pan', methods=('GET', 'POST'))
def pan():
    query = 0 ##사용자 입력 문장(없으면 0)
    t1, t2 = 0, 0 #시간 확인
    u = "https://www.law.go.kr/DRF/lawService.do?OC=ID&target=dlytrmRlt&query=" #법률용어 변환 api
    pan_list = []
    if request.method == 'POST':
        query = request.form['input']
        t1 =time.time()
        nums, documents, contents = [], [], []
        q = tokenize_sentence(query) #토큰화
        newQ = '' #토큰화된 사용자 입력을 동의어인 법률용어로 변환
        for w in q:
            if w[1] != 'NNG':
                newQ += w[0] + ' '
                continue
            w = w[0]
            a = '' #추가할 단어 저장
            w = w.replace('▁', '')
            url = u + parse.quote(w)
            try:
              xml = REQ.urlopen(url).read()
              soup = BeautifulSoup(xml, "lxml-xml")
              try:
                wList1 = soup.select('용어관계') #용어관계(동의어, 반의어, 상위어, 하위어 등) 리스트
                wList2 = soup.select('법령용어명') #법령용어 리스트
                t = 0
                for k in range(len(wList1[:5])): #관련도 높은 순으로 나오고, 시간이 오래 걸리므로 다섯개만 확인
                  if (wList1[k].text == '동의어'): #동의어라면 바꾸고 break
                    a = wList2[k].text
                    t = 1
                    break
                if(t == 0):
                  a = w
              except:
                a = w
            except:
                a = w
            newQ += a + ' '
        label = predict(newQ)
         #판례가 저장된 db에서 일련번호, 판시사항(요약 후 토큰화한 상태), 판결요지(요약한 상태) 불러오기
        db_name = 'pan'
        cnx = mysql.connector.connect(user=f'{user_name}', password=f'{pass_my}',
                              host=f'{host_my}',database=f'{db_name}')
        cursor = cnx.cursor()
        sql = '''SELECT number FROM pan WHERE label = %s;'''
        cursor.execute(sql, (str(label),))
        nums = cursor.fetchall() #일련번호 리스트
        sql = '''SELECT panyo FROM pan WHERE label = %s;'''
        cursor.execute(sql, (str(label),))
        contents = cursor.fetchall() #판결요지 리스트(사용자에게 제공)
        sql = '''SELECT pansi FROM pan WHERE label = %s;'''
        cursor.execute(sql, (str(label), ))
        documents = cursor.fetchall() #판시사항 리스트(유사도 검사)
        sql = '''SELECT * FROM pan WHERE label = %s;'''
        cnx.close()
        document_embedding_list = get_document_vectors(documents) #임베딩(판시사항)
        f2v_q = get_document_vectors([[newQ, 0]]) #임베딩(사용자 입력)
        sim_scores = [[nums[i][0], contents[i][0], cosine_similarity(f2v_q, [document_embedding_list[i]]), i] for i in range(len(document_embedding_list)) if document_embedding_list[i] is not None]
        sim_scores.sort(key=lambda x: x[2], reverse=True) #sim_scores의 각 리스트 중 세번째 요소를 정렬 기준으로.
        sim_scores = sim_scores[:5]
        for s in sim_scores:
            new_s = s[1].split('<br/>')
            pan_list.append([s[0], new_s])
    session["num_list"] = [p[0] for p in pan_list]  ##session storage에 num_list 저장
    data = {}
    data['query'] = query
    data['pan_list'] = pan_list
    t2 = time.time()
    return json.dumps(data, ensure_ascii=False)
