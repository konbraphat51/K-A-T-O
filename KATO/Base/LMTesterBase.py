from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel
import torch
import pathlib
import json

class LMParameters:
    def __init__(
        self,
        max_new_tokens:int = 64,
        do_sample:bool = True,
        temperature:float = 0.7,
        top_p:float = 0.75,
        top_k:int = 40,
        no_repeat_ngram_size:int = 2
    ):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
    def save(self, path: pathlib.Path):
        '''
        プロパティを保存する
        '''
        args = vars(self)
        
        json_data = json.dumps(args)
        with open(path / "lm_properties.json", mode='w') as f:
            f.write(json_data)

class LMTesterBase:
    '''
    できたLMを試すクラス
    '''
    
    def prepare(
        self,
        lm_model_name: str,
        tokenizer_model_name:str, 
        useint8: bool,
        peft_model_name: str = None,
    ):
        '''
        初期設定を行う。`talk()`前にこれを実行する必要がある。
        '''
        
        self.tokenizer = self.get_tokenizer(tokenizer_model_name, useint8)
        self.lm = self.get_lm(lm_model_name, useint8)
        
        if peft_model_name != None:
            self.model = self.prepare_peft(peft_model_name)
            
    def talk(
        self,
        prompt: str,
        params: LMParameters
    ) -> str:
        '''
        推論の実行。  
        パラメーター調整：https://zenn.dev/tyaahan/articles/a8d99900000002
        '''
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.lm.device)
        with torch.no_grad():
            tokens = self.lm.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=params.top_p,
                top_k=params.top_k,
                no_repeat_ngram_size=params.no_repeat_ngram_size,
            )

        output = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return output
        
    def test_talking(self, prompts, params: LMParameters):
        '''
        複数プロンプトを一気に渡す
        '''
        
        outputs = []
        for prompt in prompts:
            output = self.talk(prompt, params)
            outputs.append(output)
            
        return outputs
        
    def finish(self):
        '''
        終了処理：GPUメモリの解放
        '''
        
        del self.lm
        del self.tokenizer
        torch.cuda.empty_cache()
        
    def get_tokenizer(self, model_name, useint8):
        #モデルによって場合分け
        if model_name == "novelai/nerdstash-tokenizer-v1":
            return LlamaTokenizer.from_pretrained(model_name, load_in_8bit=useint8, device_map="auto")
        elif "line-corporation" in model_name:
            return AutoTokenizer.from_pretrained(model_name, load_in_8bit=useint8, device_map="auto", use_fast=False)
        else:
            return AutoTokenizer.from_pretrained(model_name, load_in_8bit=useint8, device_map="auto")
    
    def get_lm(self, model_name, useint8):
        #モデルによって場合分け
        if "stablelm" in model_name:
            return AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=useint8, device_map="auto", trust_remote_code=True)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=useint8, device_map="auto")
        
    def prepare_peft(self, peft_name):
        return PeftModel.from_pretrained(
            self.lm,
            peft_name,
            device_map="auto"
        )