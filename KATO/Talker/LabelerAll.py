from KATO.Base.LabelerBase import LabelerBase
import pathlib

class LabelerAll(LabelerBase):
    '''
    全ての書き起こしデータをそのまま教師データとして保存するクラス。
    '''
    
    def __init__(self) -> None:
        super().__init__(pathlib.Path(__file__).parent)
        self.data_name = "teacher_data_all"
    
    def process_transcription(self, transcriptions):
        new_list = []
        previous = ""
        #連続重複を除外する
        for text in transcriptions:
            if text != previous:
                new_list.append(text)
                previous = text
        
        return [" ".join(new_list)]
    
    def save_data(self, df_teacher_data, year):
        df_teacher_data.to_csv(self.cwd / "teacher_data_all" / (self.data_name + "_" + str(year) + ".csv"), index=True)
        return
    
if __name__ == "__main__":
    for year in range(2009, 2024):
        LabelerAll().run(year)