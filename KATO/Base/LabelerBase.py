from KATO import Consts
import pandas as pd
from tqdm import tqdm
import pathlib

class LabelerBase:
    '''
    教師データを作成する抽象クラス。
    '''
    
    def run(self, year = 2023):
        target_indicies_models = self.get_target_transcription_indicies(year)
    
        datas = []
        for index, model in tqdm(target_indicies_models):
            data = self.dig_transcription(index, model)
            datas.extend(data)
        
        df_teacher_data = self.make_teacher_data(data)
    
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
        return self.process_transcription(df_transcription[~df_transcription["text"].duplicated()]["text"])
    
    def process_transcription(self, transcription):
        '''
        書き起こしデータを処理するabstract関数。入力は書き起こしテキストのSeries
        '''
        raise NotImplementedError()
    
    def make_teacher_data(self, datas):
        '''
        教師データを作成する
        '''
        
        df_teacher_data = pd.DataFrame(datas, columns=["generated"])
        
        #シャッフル
        return df_teacher_data.sample(frac=1)
    
    def save_data(self, df_teacher_data, year):
        df_teacher_data.to_csv(pathlib.Path(__file__).parent / ("teacher_data_" + str(year) + ".csv"), index=True)
        return