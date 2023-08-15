from KATO.Prototype.Utils import Consts, Utils
from KATO.rake_ja import JapaneseRake, Tokenizer
import pandas as pd
import spacy
from spacy.symbols import nsubj, VERB
import pathlib
from tqdm import tqdm
import random

class Laberer:
    '''
    主語のある文を検知し、  
    input:「Q.(主語)についてどう思う？」  
    output: 「Q.(主語)についてどう思う？ A.(その文章)」
    の学習データを作成する
    '''
    
    SENTENCE_MIN = 10
    SENTENCE_MAX = 50
    
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.rake = JapaneseRake(max_length=3)
        self.tokenizer = Tokenizer(rawargs=Consts.mecab_params)
    
    def run(self, year = 2023):
        target_indicies_models = self.get_target_transcription_indicies(year)
    
        sub2text = []
        for index, model in tqdm(target_indicies_models):
            sub2text_this = self.dig_transcription(index, model)
            sub2text.extend(sub2text_this)
        
        df_teacher_data = self.make_teacher_data(sub2text)
    
        self.save_data(df_teacher_data, year)
    
    def get_target_transcription_indicies(self, year):
        df_video_links = pd.read_csv(Consts.data_folder / "video_links.csv")
        
        #dateがyearのものだけを抽出
        df_video_links = df_video_links[df_video_links["date"].str.contains(str(year))]
        
        #transcriptionがあるものだけを抽出
        df_video_links = df_video_links[df_video_links["transcribed"] >= 0]
        
        output = []
        for index, model in zip(df_video_links.index, df_video_links["transcribed"]):
            output.append((index, model))
        
        return output
        
    def dig_transcription(self, index, model):
        df_transcription = pd.read_csv(Consts.data_folder / "Transcription_raw" / Utils.make_transcription_file_name(index, model))
        return self.dig_sentence(df_transcription[~df_transcription["text"].duplicated()]["text"])
            
    def dig_sentence(self, texts):
        text = "".join(texts)
        if len(text) < 1e4:
            output = []  
            
            doc = self.nlp(text)
            for sent in doc.sents:
                if not self.SENTENCE_MIN <= len(sent) <= self.SENTENCE_MAX:
                    continue
                
                subject = self.get_subject(sent)
                if subject != None:
                    output.append([subject, sent.text]) 
            
            return output
        else:
            return self.dig_sentence(texts.iloc[:len(texts)//2]) + self.dig_sentence(texts.iloc[len(texts)//2:])
    
    def get_subject(self, sentence):
        '''
        主語を取得する
        '''
            
        tokens = self.tokenizer.tokenize(str(sentence))
        
        self.rake.extract_keywords_from_text(tokens)
        
        subjects = self.rake.get_ranked_phrases()
        if len(subjects) == 0:
            return None
        else:        
            return subjects[0].replace(" ", "")
    
    def make_teacher_data(self, sub2text):
        '''
        教師データを作成する
        '''
        
        questions = []
        answers = []
        for sub, text in sub2text:
            q = "Q." + sub + "についてどう思う？ A."
            a = q + text
            questions.append(q)
            answers.append(a)
        
        df_teacher_data = pd.DataFrame(zip(questions, answers), columns=["original", "generated"])
        
        return df_teacher_data.sample(frac=1)
    
    def save_data(self, df_teacher_data, year):
        df_teacher_data.to_csv(pathlib.Path(__file__).parent / ("teacher_data_" + str(year) + ".csv"), index=False)
        return
    
if __name__ == "__main__":
    laberer = Laberer()
    laberer.run(year = 2023)