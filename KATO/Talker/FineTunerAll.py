from KATO.Base import FineTunerBase, FineTunerPropertiesBase
import pathlib
from transformers import LlamaTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

class FineTunerAll(FineTunerBase):
    '''
    書き起こし文をそのまま学習するFineTuner
    '''
    
    def __init__(self):
        super().__init__(pathlib.Path(__file__).parent)
        
    def get_teacher_data_path(self):
        return self.finetuner_properties.teacher_data_path / ("teacher_data_all_"+self.finetuner_properties.year+".csv")
        
    def get_tokenizer(self):
        return LlamaTokenizer.from_pretrained(self.finetuner_properties.tokenizer_model_name, device_map="auto", load_in_8bit=self.finetuner_properties.useint8, trust_remote_code=True)
    
    def get_lm(self):
        return AutoModelForCausalLM.from_pretrained(self.finetuner_properties.lm_model_name, load_in_8bit=self.finetuner_properties.useint8, device_map="auto", trust_remote_code=True)
        
if __name__ == "__main__":
    properties = FineTunerPropertiesBase(
        lm_model_name="stabilityai/japanese-stablelm-base-alpha-7b",
        tokenizer_model_name="novelai/nerdstash-tokenizer-v1",
        year="2015",
        sample_n=100,
        ta_train_batch_size=1,
        teacher_data_path=pathlib.Path(__file__).parent / "teacher_data_all",
        output_dir=pathlib.Path(__file__).parent / "output_all"
    )
    
    finetuner = FineTunerAll()
    finetuner.run(properties)