import random
import pandas as pd
import re
import sys
import datetime
import warnings
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
       
# read the xlsx file with path and returns a list of strings
# todo: remove comma, periods, ...
def load_database(path:str):
    excel_data = pd.read_excel(path)
    data = excel_data["database"].tolist()
    return data

# shingle(text, k) returns a set of tuples containing all combinations of k consecutive words
def shingle(text:str, k:int):
    text = text.split()
    shingle_set = []
    for i in range(len(text) - k + 1):
        shingle_set.append(text[i:i+k])
    # convert lists to tuples
    for i in range(len(shingle_set)):
        shingle_set[i] = tuple(shingle_set[i])
    return set(shingle_set)

# takes in the list of text and shingle length
# returns a list of sets of tuples of words
def create_shingles(data:list, k:int):
    shingles = []
    for i in data:
        shingles.append(shingle(i, k))
    return shingles

# takes in a list of sets of tuples of words
# returns the union of all sets
def shingle_database(shingles):
    result = set([])
    for i in shingles:
        result = result.union(i)
    return result    


def create_hash_func(size: int):
    # function for creating the hash vector/function
    hash_ex = list(range(1, size + 1))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_func(vocab_size: int, nbits: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes

def create_hash(hot: list, minhash_func, shingle_union):
    # use this function for creating our signatures (eg the matching)
    signature = []
    for func in minhash_func:
        for i in range(1, len(shingle_union)+1):
            idx = func.index(i)
            signature_val = hot[idx]
            if signature_val == 1:
                signature.append(i)
                break
    return signature

def jaccard(x, y):
    return len(x.intersection(y)) / len(x.union(y))

def split_vector(signature, b):
    assert len(signature) % b == 0
    r = int(len(signature) / b)
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i : i + r])
    return subvecs

# return set of indices where the sample is similar to the database
def sim_index(sample_subvecs, sig_subvecs):
    result = set([])
    for i in range(len(sample_subvecs)):
        for j in range(len(sig_subvecs)):
            if sample_subvecs[i] == sig_subvecs[j][i]:
                result.add(j)
    return result

# return all the shingles in the sample that are also in the database 
def similar(sample, database, indices):
    result = []
    for i in indices:
        for shingle in database[i]:
            if shingle in sample:
                result.append(shingle)
    return result

# given the sample and an article, return the plag_shingle 
def plag(sample:list, article:str, k):
    article_shingle = shingle(article, k)
    sample_words = len(sample)
    plag_shingle = []
    for i in range(sample_words - k + 1):
        s = tuple(sample[i : i + k])
        if s in article_shingle:
            plag_shingle.append(s)

    return plag_shingle
    
# merge the given non-empty list of shingles
def merge_shingles(s):
    result = [s[0]]
    for i in range(1, len(s)):
        si_len = len(s[i])
        if s[i][0:si_len - 1] == result[-1][len(result[-1]) - si_len + 1:len(result[-1])]:
            temp = list(result[-1])
            temp.append(s[i][-1])
            result[-1] = tuple(temp)
        else:
            result.append(s[i])
    return result

# returns the number of words in a list of shingles
def count_words(shingles):
    num_words = 0
    for i in shingles:
        num_words += len(i)
    return num_words

# determines if s is a sub shingle of shingle
def isSubShingle(shingle, s):
    if len(s) > len(shingle):
        return False
    for i in range(len(shingle) - len(s) + 1):
        if shingle[i : i + len(s)] == s:
            return True
    return False

# determines if s is a unique shingle in shingles
def isUnique(shingles, s):
    for shingle in shingles:
        if isSubShingle(shingle, s):
            return False
    return True

# returns the starting and ending index of s in text
# s is in text
# e.g. [2, 5]  means starting 2, ending 4
def shingle_index(text:list, s:tuple):
    l = len(s)
    s = list(s)
    for i in range(len(text)):
        if text[i : i + l] == s:
            return [i, i + l]
    print(text)
    print(s)

# return the end index of run at index i
# e.g. long_run([1,1,1,1,2], 0) returns 4
def long_run(l:list, i:int):
    if (i == len(l) - 1) or (l[i + 1] != l[i]):
        return i + 1
    return long_run(l, i + 1)


########################################### pdf report ##########################################################
def color_text(sample:list, article_plag, plag_articles_idx):
    color = [-1] * len(sample)
    for shingles, c in zip(article_plag, plag_articles_idx):
        for shingle in shingles:
            index = shingle_index(sample, shingle)
            for i in range(index[0], index[1]):
                if color[i] == -1:
                    color[i] = c
    return color

def color_text_all(sample:list, article_plag, plag_articles_idx):
    color = [set([])] * len(sample)
    for shingles, c in zip(article_plag, plag_articles_idx):
        for shingle in shingles:
            index = shingle_index(sample, shingle)
            for i in range(index[0], index[1]):
                color[i] = color[i].union(set([c]))
    return color

def generate_color_text(sample:str, color:list):
    if sample.isspace():
        return sample

    words = sample.split()
    ws = re.findall("\s+", sample.strip())
    result = ""
    color_list = ["yellow", "cyan", "pink", "orange"]
    
    current_word = 0
    current_ws = 0
    while (1):
        # choose color
        c = random.choice(color_list)
        color_start = f"<font backcolor='{c}'>"
        color_end = "</font>"    
        
        end = long_run(color, current_word)
        plag = 0
        if color[current_word] != -1:
            plag = 1
        result += (color_start + "<super>" + str(color[current_word]) + "</super>") * plag + words[current_word]
        current_word += 1
        while current_word != end:
            result += ws[current_ws]
            result += words[current_word]
            current_ws += 1
            current_word += 1
        result += color_end * plag
        if current_word == len(words):
            break
        result += ws[current_ws]
        current_ws += 1
    return result

# returns [raw_paraphrased, paraphrased]
def generate_paraphrased(raw_sample:str, database_name):
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(raw_sample)
    raw_paraphrased =  ""
    paraphrased = ""
    for i in sentence_list:
        # if the sentence is empty
        if i == "":
            paraphrased += "\n"
            raw_paraphrased += "\n"
            continue

        # if the sentence has similarity smaller than 30%, use the original sentence
        if similarity(i, database_name) < 20:
            paraphrased += i + " "
            raw_paraphrased += i + " "
            continue

        s = paraphrase_sentence(i,5)
        sim = [similarity(x, database_name) for x in s]
        min_sim_idx = sim.index(min(sim))
        if (similarity(i, database_name) < sim[min_sim_idx]):
            paraphrased += i + " "
            raw_paraphrased += i + " "
        else:
            paraphrased += "<font backcolor='yellow'>" + s[min_sim_idx] + "</font>" + " "
            raw_paraphrased += s[min_sim_idx] + " "
    return [raw_paraphrased, paraphrased]


def report(file_name, database_name, similarity_index, articles_similarity, report_text, para_text):
    # styles
    pdf_name = 'report.pdf'
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='centered', alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='index', alignment=TA_CENTER, textColor=colors.blue, fontSize=24, leading=24*1.2))
    styleN = styles['Normal']
    styleH = styles['Heading1']
    story = []
    doc = SimpleDocTemplate(
        pdf_name,
        pagesize=letter,
        bottomMargin=.4 * inch,
        topMargin=.6 * inch,
        rightMargin=.8 * inch,
        leftMargin=.8 * inch)
    
    # contents
    paraphrased_sim = 0
    total_words = 0
    for p in para_text:
        pwords = len(p[0].split())
        total_words += pwords
        paraphrased_sim += similarity(p[0], database_name) * pwords
    paraphrased_sim /= total_words
    paraphrased_sim = round(paraphrased_sim, 1)     
    
    name = Paragraph("<para alignment='right'>" + file_name + "</para>", styleN)
    date = datetime.datetime.now()
    date = Paragraph("<para alignment='right'>" + date.strftime("%b %d, %Y") + "</para>", styleN)
    
    heading = Paragraph("<para fontSize=26 leading=30> \
                                    Similarity Index for " + file_name + \
                        "</para>", styleH)  
    sources_overview = Paragraph("<para fontName='times-roman' fontSize=14 leading=20> \
                                    <b>Sources Overview</b> \
                                  </para>", styleN)  
    index = Paragraph(str(similarity_index) + "%", styles["index"])
    overall_similarity = Paragraph("<para fontSize=8 alignment='center'> \
                           <b>OVERALL SIMILARITY</b> \
                         </para>")
    text_heading = Paragraph("<para fontSize=16 leading=19>" + 
                                    file_name + 
                                   "</para>", styleH)
    paraphrase_heading = Paragraph("<para fontSize=16 leading=19>" + 
                                    "Recommended Paraphrase" + 
                                   "</para>", styleH)
    paraphrase_index = Paragraph(str(paraphrased_sim) + "%", styles["index"])

    story.append(name)
    story.append(date)
    story.append(heading)
    story.append(sources_overview)
    story.append(index)
    story.append(overall_similarity)
    
    
    # similarity for each article
    for i in range(len(articles_similarity)):
        if articles_similarity[i] == 0:
            continue
        p1 = Paragraph("<para fontSize=16 leading=0>" + 
                         str(i) +
                      "</para>", styleN)
        p2 = Paragraph("<para fontSize=16 leading=25 alignment='right'color='blue'>" + 
                         str(articles_similarity[i]) + "%" +
                      "</para>", styleN)
        story.append(p1)
        story.append(p2)
    

    story.append(PageBreak())

    # original text page
    story.append(text_heading)
    story.append(index)
    story.append(overall_similarity)

    for i in report_text:
        p = Paragraph("<para fontSize=12 leading=17>" + i + "</para>", styleN)
        story.append(p)
   
    story.append(PageBreak())

    # paraphrase page
    story.append(paraphrase_heading)
    story.append(paraphrase_index)
    story.append(overall_similarity)

    for i in para_text:
        p = Paragraph("<para fontSize=12 leading=17>" + i[1] + "</para>", styleN)
        story.append(p)

    story.append(PageBreak())

    # paraphrased color text
    story.append(Spacer(20, 20))
    story.append(sources_overview)
    story.append(Spacer(10, 10))

    para_colored = []
    for p in para_text:
        c = color(p[0], database_name)[0]
        para_colored.append(generate_color_text(p[0], c))

    for i in para_colored:
        p = Paragraph("<para fontSize=12 leading=17>" + i + "</para>", styleN)
        story.append(p)

    doc.build(story,) 


########################################### paraphrase ###################################################

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def paraphrase_sentence(input_text,num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def paraphrase_paragraph(paragraph):
    paraphrase = []
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(paragraph)
    for i in sentence_list:
        a = paraphrase_sentence(i,1)
        paraphrase.append(a)    
    paraphrase2 = [' '.join(x) for x in paraphrase]
    paraphrase3 = [' '.join(x for x in paraphrase2)]
    paraphrased_text = str(paraphrase3).strip('[]').strip("'")
    return paraphrased_text

##########################################################################################################

def similarity(raw_sample, database_name):
    k = 5
    data = load_database(database_name)
    # convert to lowercase, remove punctuations
    for i in range(len(data)):
        data[i] = data[i].lower()
        data[i] = re.sub(r'[^\w\s-]', ' ', data[i])
    
    # create shingle database
    shingles = create_shingles(data, k)
    
    # variables
    sample = raw_sample.lower()
    sample = sample.split()
    sample = [re.sub(r'[^\w\s-]', ' ', x) for x in sample] 
    sample_words = len(sample)
    if sample_words == 0:
        return 0
    plag_articles_idx = []   
    plag_index = []
                
    for article_idx in range(len(shingles)):
        for i in range(sample_words - k + 1):
            shingle = tuple(sample[i : i + k])
            if (shingle in shingles[article_idx]) and not (article_idx in plag_articles_idx):
                plag_articles_idx.append(article_idx)
            
            
    # find similar words (get article_plag)
    article_plag = []
    for article_idx in plag_articles_idx:
        article = data[article_idx]
        temp = plag(sample, article, 3)
        article_plag.append(temp)
    
    # color the text
    color = color_text(sample, article_plag, plag_articles_idx)   
    
    similarity_index = round((sample_words - color.count(-1)) / sample_words * 100, 1)
    return similarity_index

# return [color, color_all]
def color(raw_sample, database_name):
    k = 5
    data = load_database(database_name)
    # convert to lowercase, remove punctuations
    for i in range(len(data)):
        data[i] = data[i].lower()
        data[i] = re.sub(r'[^\w\s-]', ' ', data[i])
    
    # create shingle database
    shingles = create_shingles(data, k)
    
    # variables
    sample = raw_sample.lower()
    sample = sample.split()
    sample = [re.sub(r'[^\w\s-]', ' ', x) for x in sample] 
    sample_words = len(sample)
    plag_articles_idx = []   
    plag_index = []
                
    for article_idx in range(len(shingles)):
        for i in range(sample_words - k + 1):
            shingle = tuple(sample[i : i + k])
            if (shingle in shingles[article_idx]) and not (article_idx in plag_articles_idx):
                plag_articles_idx.append(article_idx)
            
            
    # find similar words (get article_plag)
    article_plag = []
    for article_idx in plag_articles_idx:
        article = data[article_idx]
        temp = plag(sample, article, 3)
        article_plag.append(temp)
    
    # color the text
    color = color_text(sample, article_plag, plag_articles_idx) 
    color_all = color_text_all(sample, article_plag, plag_articles_idx) 

    return [color, color_all]

def sim(file_name, raw_sample, database_name):
    data = load_database(database_name)
    paragraphs = raw_sample.splitlines()
    sample = raw_sample.lower()
    sample = sample.split()
    sample = [re.sub(r'[^\w\s-]', ' ', x) for x in sample] 

    sample_words = len(sample)

    # obtain colors
    paragraph_color = []
    paragraph_color_all = []
    for p in paragraphs:
        c = color(p, database_name)
        paragraph_color.append(c[0])
        paragraph_color_all.append(c[1])

    # calculate similarity for each article
    articles_similarity = [0] * len(data)
    for p in paragraph_color_all:
        for word in p:
            for c in word:
                articles_similarity[c] += 1
    articles_similarity = [round(s / sample_words * 100, 1) for s in articles_similarity]   

    # generate report text
    report_text = []
    for i in range(len(paragraphs)):
        t = generate_color_text(paragraphs[i], paragraph_color[i])
        report_text.append(t)
    
    # generate paraphrase
    # para_text [[raw_para, para], ...]
    para_text = []
    for i in range(len(paragraphs)):
        t = generate_paraphrased(paragraphs[i], database_name)
        para_text.append(t)
    para = generate_paraphrased(raw_sample, database_name)

    # calculate overall similarity index
    count = 0
    for p in paragraph_color:
        count += p.count(-1)
    similarity_index = round((sample_words - count) / sample_words * 100, 1)

    # create report pdf
    report(file_name, database_name, similarity_index, articles_similarity, report_text, para_text)  


def simple(file_name, raw_sample, database_name):
    k = 5
    data = load_database(database_name)
    # convert to lowercase, remove punctuations
    for i in range(len(data)):
        data[i] = data[i].lower()
        data[i] = re.sub(r'[^\w\s-]', ' ', data[i])
    
    # create shingle database
    shingles = create_shingles(data, k)
    
    # variables
    sample = raw_sample.lower()
    sample = sample.split()
    sample = [re.sub(r'[^\w\s-]', ' ', x) for x in sample] 
    sample_words = len(sample)
    plag_articles_idx = []   
    plag_index = []
    
    # find similar articles (get plag_articles_idx)              
    for article_idx in range(len(shingles)):
        for i in range(sample_words - k + 1):
            shingle = tuple(sample[i : i + k])
            if (shingle in shingles[article_idx]) and not (article_idx in plag_articles_idx):
                plag_articles_idx.append(article_idx)
            
            
    # find similar words (get article_plag)
    article_plag = []
    for article_idx in plag_articles_idx:
        article = data[article_idx]
        temp = plag(sample, article, 3)
        article_plag.append(temp)
    
    # color the text
    color = color_text(sample, article_plag, plag_articles_idx)  
    color_all = color_text_all(sample, article_plag, plag_articles_idx)  

    # calculate similarity for each article
    articles_similarity = [0] * len(plag_articles_idx)
    for word in color_all:
        for c in word:
            idx = plag_articles_idx.index(c)
            articles_similarity[idx] += 1
    articles_similarity = [round(s / sample_words * 100, 1) for s in articles_similarity]   
    
    # generate report text
    report_text = generate_color_text(raw_sample, color)

    # generate paraphrased text
    para = generate_paraphrased(raw_sample, database_name)
    
    # create report pdf
    similarity_index = round((sample_words - color.count(-1)) / sample_words * 100, 1)
    report(file_name, database_name, similarity_index, plag_articles_idx, articles_similarity, report_text, para[0], para[1])  

def main():
    warnings.simplefilter("ignore")
    try:
        file_name = sys.argv[1]
        database_name = sys.argv[2]
        f = open(file_name, "r")

    except:
        print("Incorrect number of arguments\n")
        print("Usage: python similarity.py <file-name>.txt <database-name>.xlsx")

    else:
        sim(file_name, f.read(), database_name)


if __name__ == "__main__":
    main()





