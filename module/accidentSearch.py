import pandas as pd

class AccidentSearch():
    def __init__(self):
        self.df = pd.read_csv('./files/Incident_Type_Classification_Table.csv',encoding="cp949")
    
    def select_type_num(self, type_num):
        temp_df = self.df.loc[self.df["사고유형"]==type_num]
        result = temp_df.to_dict(orient='records')
        ##Key: '사고객체', '사고장소', '사고장소특징', 'A진행방향', 'B진행방향', '과실비율A', '과실비율B', '사고유형'
        return result
        

if __name__=="__main__":
    ads = AccidentSearch()
    print(ads.select_type_num(3))