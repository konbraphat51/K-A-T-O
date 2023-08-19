from KATO.Base import FineTunerBase, FineTunerPropertiesBase
import pathlib

class FineTunerAll(FineTunerBase):
    '''
    書き起こし文をそのまま学習するFineTuner
    '''
    
    def __init__(self):
        super().__init__(pathlib.Path(__file__).parent)
        
    def get_teacher_data_path(self):
        return self.finetuner_properties.teacher_data_path / ("teacher_data_all_"+self.finetuner_properties.year+".csv")
        
if __name__ == "__main__":
    properties = FineTunerPropertiesBase(
        lm_model_name="stabilityai/japanese-stablelm-base-alpha-7b",
        tokenizer_model_name="novelai/nerdstash-tokenizer-v1",
        year="2015",
        sample_n=30,
        ta_train_batch_size=1,
        teacher_data_path=pathlib.Path(__file__).parent / "teacher_data_all",
        output_dir=pathlib.Path(__file__).parent / "output_all"
    )
    
    finetuner = FineTunerAll()
    finetuner.run(properties)