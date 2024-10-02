import math

class HaversineDistanceClassifier:
    def __init__(self, lat1, lon1, lat2, lon2):
        """
        초기화 메서드. 시작점과 끝점의 위도, 경도를 받습니다.
        """
        self.lat1 = math.radians(lat1)
        self.lon1 = math.radians(lon1)
        self.lat2 = math.radians(lat2)
        self.lon2 = math.radians(lon2)
        self.R = 6371.0  # 지구의 반지름 (킬로미터 단위)

    def calculate_distance(self):
        """
        Haversine 공식을 사용해 두 지점 간의 거리를 계산하는 메서드.
        """
        # 위도, 경도 차이 계산
        dlon = self.lon2 - self.lon1
        dlat = self.lat2 - self.lat1
        
        # Haversine 공식 적용
        a = math.sin(dlat / 2)**2 + math.cos(self.lat1) * math.cos(self.lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # 거리 계산 (킬로미터로 반환)
        distance_km = self.R * c
        return distance_km * 1000  # 미터로 변환

    def classify_distance(self):
        """
        거리 범위에 따라 클래스를 반환하는 메서드.
        - 250m <= 거리 < 500m: class 0
        - 500m <= 거리 < 1500m: class 1
        - 1500m 이상: class 2
        """
        distance = self.calculate_distance()

        if 250 <= distance < 500:
            return 0
        elif 500 <= distance < 1500:
            return 1
        elif distance >= 1500:
            return 2
        else:
            return None  # 250m 미만은 클래스 없음

# 사용 예시
if __name__ == "__main__":
    # 서울 (위도, 경도) -> 도쿄 (위도, 경도)
    lat1, lon1 = 37.5665, 126.9780  # 서울
    lat2, lon2 = 37.5645, 126.9800  # 서울 근처 다른 지점
    
    distance_classifier = HaversineDistanceClassifier(lat1, lon1, lat2, lon2)
    
    # 거리 계산
    distance = distance_classifier.calculate_distance()
    print(f"두 지점 사이의 거리: {distance:.2f} m")
    
    # 클래스 분류
    classification = distance_classifier.classify_distance()
    print(f"거리에 따른 클래스: {classification}")