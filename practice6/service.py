import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# === Hyper Parameter === #

# Ridge 모델 사용, 
# alpha=10, polyDegree=2
DEGREE = 3
ALPHA = 10

# ======================= #

class PredictService :
    def __init__(self):
        self.poly = PolynomialFeatures(degree = DEGREE, include_bias=False)
        self.ss = StandardScaler()
        self.ridge = Ridge(alpha=ALPHA)
        self.features = [
            'avgFuelEfficiency',
            'cumulativeMileageKm',
            'elapsedShipmentMonth',
            'accidentDepreciationCount',
            'ownershipChangesCount'
        ]
        self.target = 'salePrice'

    def trainModel(self, data):
        # 데이터베이스에 저장된 전체 데이터를 기반으로 예측 모델을 (재)학습하고 최신화

        # [1] df 가져오기
        df = pd.DataFrame(data)

        # [2] 학습데이터/정답데이터 분리하고, train/test 나누기
        df_full = df[self.features]
        df_target = df[self.target]

        train_input, test_input, train_target, test_target = train_test_split(df_full, df_target, test_size=0.2, random_state=42)

        # Poly : 특성을 늘리는 단계
        self.poly.fit(train_input)
        train_poly = self.poly.transform(train_input)
        test_poly = self.poly.transform(test_input)

        # Scaling : 스케일을 일정한 단위로 맞추는 단계
        self.ss.fit(train_poly)
        train_scaled = self.ss.transform(train_poly)
        test_scaled = self.ss.transform(test_poly)

        # fit : 학습 단계
        self.ridge.fit(train_scaled, train_target)

        # 결정계수 계산 (모델 테스트)
        r2 = self.ridge.score(test_scaled, test_target)  # r2 : 결정계수(점수) , 0~1
        
        return r2

    def predict(self, variables):
        # 5개 핵심 변수(평균 연비, 누적 주행거리(km), 출고 후 경과 월수, 사고 감가 건수, 소유자 변경 횟수*)를 모델에 넣고, 예측한 매매 가격(단위: 만 원)을 실시간으로 반환
        # {
        #     "avgFuelEfficiency": 11.5,
        #     "cumulativeMileageKm": 15000,
        #     "elapsedShipmentMonth": 14,
        #     "accidentDepreciationCount": 0,
        #     "ownershipChangesCount": 0
        # }

        request_df = pd.DataFrame([variables])
        request_df = request_df[self.features]  # 순서 맞추는 용도.
        request_poly = self.poly.transform(request_df)
        request_scaled = self.ss.transform(request_poly)
        result = self.ridge.predict(request_scaled)     # 이건 넘파이배열로 반환됨. 이걸 인덱스로 추출해야 함
        return result[0]

predictService = PredictService()