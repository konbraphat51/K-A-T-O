from KATO import Consts, Utils
import pandas as pd
from tqdm import tqdm
import pathlib

class LabelerBase:
    '''
    教師データを作成する抽象クラス。
    '''
    
    def __init__(self, cwd: pathlib.Path) -> None:
        self.cwd = cwd
    
    def run(self, year: int):
        target_indicies_models = self.get_target_transcription_indicies(year)
    
        datas = []
        for index, model in tqdm(target_indicies_models):
            data = self.dig_transcription(index, model)
            datas.extend(data)
        
        df_teacher_data = self.make_teacher_data(datas)
    
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
        
        return self.process_transcription(df_transcription["text"].dropna())
    
    def process_transcription(self, transcriptions):
        '''
        書き起こしデータを処理するabstract関数。  
        入力は書き起こしテキストのSeries  
        出力は生成文のリスト
        '''
        raise NotImplementedError()
    
    def make_teacher_data(self, datas):
        '''
        教師データを作成する
        '''
        
        df_teacher_data = pd.DataFrame(datas, columns=["text"])
        
        #シャッフル
        return df_teacher_data.sample(frac=1)
    
    def save_data(self, df_teacher_data, year):
        df_teacher_data.to_csv(self.cwd / (self.data_name + "_" + str(year) + ".csv"), index=True)
        return