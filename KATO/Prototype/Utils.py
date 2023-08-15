import pathlib

class Consts:
    data_folder = "KATO\\Prototype\\Data\\"
    mecab_params = '-r "C:/Program Files/MeCab/etc/mecabrc" -u "C:/Program Files/MeCab/dic/unidic_kato.dic"'

    
Consts.data_folder = pathlib.Path(__file__).parent.parent / "Data"

class Utils:
    def make_transcription_file_name(index, model):
        return str(index) + "-" + str(model) + ".csv"