from KATO.Base import LMTester, LMParameters

class LMTesterAll(LMTester):
    pass

if __name__ == '__main__':
    lmtester = LMTesterAll()
    
    parameters = LMParameters(
        max_new_tokens = 64,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.75,
        top_k = 40,
        no_repeat_ngram_size = 2
    )
    
    lmtester.prepare(
        lm_model_name = "microsoft/DialoGPT-small",
        tokenizer_model_name = "microsoft/DialoGPT-small",
        useint8 = False,
        peft_model_name = None
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