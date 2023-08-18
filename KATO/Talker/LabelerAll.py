from KATO.Base.LabelerBase import LabelerBase

class LabelerAll(LabelerBase):
    '''
    全ての書き起こしデータをそのまま教師データとして保存するクラス。
    '''
    
    def __init__(self) -> None:
        super().__init__()
        self.data_name = "teacher_data_all"
    
    def process_transcription(self, transcriptions):
        return " ".join(transcriptions)
    
if __name__ == "__main__":
    for year in range(2009, 2024):
        LabelerAll().run(year)