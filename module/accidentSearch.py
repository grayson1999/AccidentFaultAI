import pandas as pd

class AccidentSearch():
    def __init__(self):
        self.df = pd.read_csv('/AccidentFaultAI/files/Incident_Type_Classification_Table.csv',encoding="cp949")
    
    def select_type_num(self, type_num):
        temp_df = self.df.loc[self.df["AccidentType"]==type_num]
        result = temp_df.to_dict(orient='records')
        ##Key: AccidentObject,AccidentLocation,AccidentLocationCharacteristics,DirectionOfA,DirectionOfB,FaultRatioA,FaultRatioB,AccidentType
        return result
        
    def get_all_data(self):
        return pd.DataFrame(self.df)

if __name__=="__main__":
    ads = AccidentSearch()
    result = ads.select_type_num(3)
    print(result)