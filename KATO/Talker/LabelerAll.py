from KATO.Base.LabelerBase import LabelerBase
import pathlib

class LabelerAll(LabelerBase):
    '''
    全ての書き起こしデータをそのまま教師データとして保存するクラス。
    '''
    
    def __init__(
        self, 
        max_text_length = 1000, #一つの文字列が最大何文字格納されるか 
        proceedings = 3     #文字列数増加時に、前回のリストの何テキスト文を重複して含めるか
    ) -> None:
        super().__init__(pathlib.Path(__file__).parent)
        self.data_name = "teacher_data_all"
        self.max_text_length = max_text_length
        self.proceedings = proceedings
    
    def process_transcription(self, transcriptions):
        text_list_list = []
        #連続重複を除外する
        current_length = 0
        text_list_current = [""]
        for text in transcriptions:
            if text != text_list_current[-1]:
                length = len(text)
                if current_length + length <= self.max_text_length:
                    text_list_current.append(text)
                    current_length += length
                else:
                    text_list_list.append(text_list_current)
                    text_list_current = text_list_current[-self.proceedings:]
                    text_list_current.append(text)
                    current_length = 0
                    for _text in text_list_current:
                        current_length += len(_text)
                        
        text_list_list.append(text_list_current)
        
        output_list = []
        
        for text_list in text_list_list:
            output_list.append(" ".join(text_list))
        
        return output_list
    
    def save_data(self, df_teacher_data, year):
        df_teacher_data.to_csv(self.cwd / "teacher_data_all" / (self.data_name + "_" + str(year) + ".csv"), index=True)
        return
    
if __name__ == "__main__":
    for year in range(2009, 2024):
        LabelerAll().run(year)