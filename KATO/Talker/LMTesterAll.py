from KATO.Base import LMTesterBase, LMParameters

class LMParametersAll(LMParameters):
    def __init__(
        self,
        max_new_tokens:int = 64,
        do_sample:bool = True,
        temperature:float = 0.7,
        top_p:float = 0.75,
        top_k:int = 40,
        no_repeat_ngram_size:int = 2,
        chains:int = 1,
        chain_depth:int = 1
    ):
        super().__init__(
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            no_repeat_ngram_size
        )
        self.chains = chains
        self.chain_depth = chain_depth
    

class LMTesterAll(LMTesterBase):
    def talk(
        self,
        prompt: str,
        params: LMParametersAll,
    ):
        '''
        推論の実行。
        トークのチェイン（前回の発言内容を再帰的に使用）を行う。
        '''
        
        output = super().talk(prompt, params)
        for cnt in range(params.chains-1):
            prompt_new = " ".join(output.split(" ")[-params.chain_depth:])
            output_new = super().talk(prompt_new, params)
            output += output_new[len(prompt_new):]
            
        return output

if __name__ == '__main__':
    lmtester = LMTesterAll()
    
    parameters = LMParametersAll(
        max_new_tokens = 200,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.75,
        top_k = 40,
        no_repeat_ngram_size = 2,
        chains = 5,
        chain_depth = 4,
    )
    
    lmtester.prepare(
        lm_model_name = "cyberagent/open-calm-3b",
        tokenizer_model_name = "cyberagent/open-calm-3b",
        useint8 = True,
        peft_model_name = "konbraphat51/KATO_talker_202308230014"
    )
    
    start_list = [
        "こんにちは",
        "やあ",
        "喉痛いわ",
        "今週のキングダム見た？",
        "ヒカキンは神だと思わない？"
    ]
    
    outputs = lmtester.test_talking(start_list, parameters)
    
    for output in outputs:
        print(output)
        
    lmtester.finish()