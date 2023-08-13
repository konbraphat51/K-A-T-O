from KATO.Prototype.Utils import Consts, Utils
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
    
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
    
    def run(self, year = 2015, sample_n = 30):
        target_indicies_models = self.get_target_transcription_indicies(year)

        #sample_n個だけ抽出
        target_indicies_models = random.sample(target_indicies_models, sample_n)
    
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
        
        output = []
        
        for text in df_transcription["text"]:    
            doc = self.nlp(text)
            for sent in doc.sents:
                subject = self.get_subject(sent)
                if subject != None:
                   output.append([subject, sent.text]) 
        
        return output
    
    def get_subject(self, sentence):
        '''
        主語を取得する
        '''
        
        for token in sentence:
            if token.dep == nsubj:
                return token.text
            
            
        return None
    
    def make_teacher_data(self, sub2text):
        '''
        教師データを作成する
        '''
        
        questions = []
        answers = []
        for sub, text in sub2text:
            q = "Q." + sub + "についてどう思う？"
            a = q + " A." + text
            questions.append(q)
            answers.append(a)
        
        df_teacher_data = pd.DataFrame(zip(questions, answers), columns=["original", "generated"])
        
        return df_teacher_data
    
    def save_data(self, df_teacher_data, year):
        df_teacher_data.to_csv(pathlib.Path(__file__).parent / ("teacher_data_" + str(year) + ".csv"), index=False)
        return
    
if __name__ == "__main__":
    laberer = Laberer()
    laberer.run()