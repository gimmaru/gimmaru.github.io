---
layout: post
title:  "[저평가 분석] 전자공시시스템 다트 오픈 API, OPEN Dart 사용법"
subtitle:   "부제"
categories: project
tags: undervalue
comments: true

---

오픈 다트 API를 사용하기 위해 먼저 API 인증키를 발급받아야 한다. [오픈 다트 사이트](https://opendart.fss.or.kr/)에 접속하고 인증키 신청을 누른 뒤, 몇 가지 약관에 동의하고 회원가입과 인증 절차를 거치면 바로 발급받을 수 있다. 발급받은 인증키는 인증키 신청/관리 텝의 [오픈API 이용현황](https://opendart.fss.or.kr/mng/apiUsageStatusView.do)에서 확인할 수 있으며 [`keyring`](https://pypi.org/project/keyring/) 라이브러리를 이용하면 다음과 같이 API 인증키를 필요할 때 편하게 꺼내 쓸 수 있다.

```python
import keyring

keyring.set_password("system", "username", "password")
keyring.get_password("system", "username")
```

개발가이드 텝을 보면 오픈 다트 API를 통해 사용할 수 있는 API와 각 API를 통해 얻을 수 있는 정보가 무엇인지 확인할 수 있다.

![개발가이드]({{ site.baseurl }}/assets/img/2024-02-13-project-undervalue-open_dart_api-dev_guide.png)
<center><b><a href="https://opendart.fss.or.kr/guide/main.do?apiGrpCd=DS001" target="_black">개발가이드</a></b></center>
<br />

## 고유번호 불러오기

고유번호 API 개발가이드를 살펴보자. 오픈 다트 API에서는 각 기업을 식별할 때 기업 고유번호를 사용한다. 기업 고유번호는 고유번호 API로 확인할 수 있으므로 다른 API를 사용하기 위해 먼저 알아둘 필요가 있다.

![고유번호 개발가이드]({{ site.baseurl }}/assets/img/2024-02-13-project-undervalue-open_dart_api-dev_guide_corp_code.png)
<center><b><a href="https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018" target="_black">고유번호 개발가이드</a></b></center>
<br />

기본 정보를 보면 메서드, 요청URL, 인코딩 방식, 출력포멧이 무엇인지 확인할 수 있다. 요청URL에 GET 메서드를 활용해서 UTF-8로 인코딩되어 있는 기업 고유번호를 zip 파일로 받을 수 있다는 의미이다.

요청 인자의 요청키를 보면 'crtfc_key'이고 설명을 보면 발급받은 인증키라고 한다.

이러한 설명을 참고하여 고유번호 API를 통해 기업 고유번호를 받아보자.


```python
import keyring
from typing import Optional
import pprint
import requests
from io import BytesIO
import zipfile
import xmltodict
import json
import pandas as pd

DART_API_KEY = keyring.get_password("open_dart_api", "gimmaru")

def get_corp_code(DART_API_KEY: str) -> pd.DataFrame:
    corp_code_url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    corp_code_byte = requests.get(corp_code_url)
    assert corp_code_byte.status_code == 200, f"{corp_code_byte.status_code} HTTP Error\n\n See the following sites: https://httpstatusdogs.com/"

    corp_code_zip = BytesIO(corp_code_byte.content)
    
    if zipfile.is_zipfile(corp_code_zip):
        corp_code_unzip = zipfile.ZipFile(corp_code_zip)
        corp_code_xml = corp_code_unzip.read('CORPCODE.xml').decode('utf-8')
        corp_code_dict = xmltodict.parse(corp_code_xml)
        return pd.DataFrame(corp_code_dict['result']['list'])
    else:
        status_code = xmltodict.parse(corp_code_byte.content)['result']['status']
        assert status_code == '000', f'{status_code} Error\n\nCheck Message Explanation: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018'
```

요청URL의 뒷 부분에 '?'를 붙이고 요청키와 발급받은 다트 API 키를 넘겨주었다. [URL에서 ?](https://znos.tistory.com/30)는 GET 방식으로 서버에 데이터를 요청할 때 사용된다. ? 뒤에서부터 요청할 데이터가 작성된다는 의미이다.

출력포멧이 'Zip File(binary)'이므로 바이트 타입인 zip으로 압축된 파일을 반환된다. 바이트 타입의 `raw_corp_code.content` 파일을 zipfile로 처리할 수 있도록 BytesIO를 통해 zip 파일로 변환해주고

zip 파일의 압축을 해제하고 나온 'CORPCODE.xml' 파일이 utf-8로 인코딩되었으므로 그에 맞춰 디코딩해줬다.

그 후 xmltodict을 통해 xml 파일을 파이썬 딕셔너리로 변환해주고 최종적으로 데이터프레임 형태로 반환하였다.

반환된 데이터프레임의 'corp_code' 열의 값이 고유번호를 의미한다.


```python
df_code = get_corp_code(DART_API_KEY)
df_code.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 104025 entries, 0 to 104024
    Data columns (total 4 columns):
     #   Column       Non-Null Count   Dtype 
    ---  ------       --------------   ----- 
     0   corp_code    104025 non-null  object
     1   corp_name    104025 non-null  object
     2   stock_code   3697 non-null    object
     3   modify_date  104025 non-null  object
    dtypes: object(4)
    memory usage: 3.2+ MB



```python
df_code.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corp_code</th>
      <th>corp_name</th>
      <th>stock_code</th>
      <th>modify_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>98316</th>
      <td>01748888</td>
      <td>엠케이트레이딩</td>
      <td>None</td>
      <td>20230502</td>
    </tr>
    <tr>
      <th>102744</th>
      <td>01777941</td>
      <td>하이클래스디벨롭</td>
      <td>None</td>
      <td>20230801</td>
    </tr>
    <tr>
      <th>2348</th>
      <td>00343987</td>
      <td>오떼마찌엘티디</td>
      <td>None</td>
      <td>20170630</td>
    </tr>
    <tr>
      <th>50088</th>
      <td>01506149</td>
      <td>남악개발대부</td>
      <td>None</td>
      <td>20201014</td>
    </tr>
    <tr>
      <th>95865</th>
      <td>01759510</td>
      <td>에스엘씨엔씨</td>
      <td>None</td>
      <td>20230510</td>
    </tr>
  </tbody>
</table>
</div>



결과를 확인해보면 고유번호 API를 통해 십만여개 기업에 대한 고유번호를 얻었음을 알 수 있다.

그런데 'stock_code' 열을 보면 None 값인 경우가 많다. 이러한 경우 해당 기업이 상장되지 않았음을 의미한다. 데이터프레임의 info를 보면 None인 경우도 null 값으로 처리되고 있다. None도 null 값으로 간주되므로 데이터프레임의 dropna 메서드로 비상장기업을 제거할 수 있다.


```python
df_code = (
    df_code
    .dropna()
    .reset_index(drop=True)
)

df_code
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corp_code</th>
      <th>corp_name</th>
      <th>stock_code</th>
      <th>modify_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00260985</td>
      <td>한빛네트</td>
      <td>036720</td>
      <td>20170630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00264529</td>
      <td>엔플렉스</td>
      <td>040130</td>
      <td>20170630</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00358545</td>
      <td>동서정보기술</td>
      <td>055000</td>
      <td>20170630</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00231567</td>
      <td>애드모바일</td>
      <td>032600</td>
      <td>20170630</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00247939</td>
      <td>씨모스</td>
      <td>037600</td>
      <td>20170630</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3692</th>
      <td>00126414</td>
      <td>삼성제약</td>
      <td>001360</td>
      <td>20231005</td>
    </tr>
    <tr>
      <th>3693</th>
      <td>00116426</td>
      <td>코센</td>
      <td>009730</td>
      <td>20231205</td>
    </tr>
    <tr>
      <th>3694</th>
      <td>00107987</td>
      <td>남해화학</td>
      <td>025860</td>
      <td>20231205</td>
    </tr>
    <tr>
      <th>3695</th>
      <td>01068658</td>
      <td>디딤이앤에프</td>
      <td>217620</td>
      <td>20231206</td>
    </tr>
    <tr>
      <th>3696</th>
      <td>00222213</td>
      <td>피케이엘</td>
      <td>039870</td>
      <td>20231206</td>
    </tr>
  </tbody>
</table>
<p>3697 rows × 4 columns</p>
</div>



## 상장기업 재무정보 불러오기

이렇게 가져온 기업 고유번호로 상장기업 재무정보를 가져올 수 있다. 개발가이드의 상장기업 재무정보를 통해 확인할 수 있는 오픈 다트의 API는 다음과 같다.

![개발가이드 상장기업 재무정보 목록]({{ site.baseurl }}/assets/img/2024-02-13-project-undervalue-open_dart_api-corp_fs_list.png)
<center><b><a href="https://opendart.fss.or.kr/guide/main.do?apiGrpCd=DS003" target="_black">개발가이드 상장기업 재무정보 목록</a></b></center>
<br />

그 중 단일회사 전체 재무제표를 불러와 보자.

![단일회사 전체 재무제표 개발가이드]({{ site.baseurl }}/assets/img/2024-02-13-project-undervalue-open_dart_api-corp_fs.png)
<center><b><a href="https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019020" target="_black">단일회사 전체 재무제표 개발가이드</a></b></center>
<br />

단일회사 전체 재무제표 개발가이드의 기본 정보를 보면 요청URL이 2개다. 각각 출력포멧이 'json', 'xml'이라는 점이 차이가 있다.

요청인자의 요청키를 보면 'API 인증키', '고유번호', '사업연도', '보고서 코드', '개별/연결구분'이 있다. 해당 값들을 바꿔 전달하면서 원하는 기업과 시기에 맞는 재무정보를 불러올 수 있으며, 2015년 이후 정보에 한해서 제공하고 있다는 점은 유의해야 한다.

예시로 확인해볼 기업은 삼성제약이다. 삼성제약의 고유번호는 다음과 같이 확인할 수 있다.


```python
print( df_code.loc[df_code['corp_name'] == '삼성제약', 'corp_code'].values[0] )
```

    00126414



```python
def get_financial_statements(
    DART_API_KEY: str, 
    CORP_CODE: str, 
    BSNS_YEAR: str, 
    REPRT_CODE: str, 
    FS_DIV: str
) -> pd.DataFrame:

    fs_url = f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?crtfc_key={DART_API_KEY}&corp_code={CORP_CODE}&bsns_year={BSNS_YEAR}&reprt_code={REPRT_CODE}&fs_div={FS_DIV}"
    fs_raw = requests.get(fs_url)
    assert fs_raw.status_code == 200, f"{fs_raw.status_code} HTTP Error\n\n See the following sites: https://httpstatusdogs.com/"

    fs_dict = fs_raw.json()
    assert fs_dict['status'] == '000', f"{fs_dict['status']} Error\n\n {fs_dict['message']}"

    return pd.DataFrame(fs_dict['list'])

df_fs = get_financial_statements(
    DART_API_KEY=DART_API_KEY,
    CORP_CODE = df_code.loc[df_code['corp_name'] == '삼성제약', 'corp_code'].values[0],
    BSNS_YEAR = '2015',
    REPRT_CODE = '11011',
    FS_DIV = 'CFS',
)
```


```python
df_fs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rcept_no</th>
      <th>reprt_code</th>
      <th>bsns_year</th>
      <th>corp_code</th>
      <th>sj_div</th>
      <th>sj_nm</th>
      <th>account_id</th>
      <th>account_nm</th>
      <th>account_detail</th>
      <th>thstrm_nm</th>
      <th>thstrm_amount</th>
      <th>frmtrm_nm</th>
      <th>frmtrm_amount</th>
      <th>bfefrmtrm_nm</th>
      <th>bfefrmtrm_amount</th>
      <th>ord</th>
      <th>currency</th>
      <th>thstrm_add_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_NoncurrentAssets</td>
      <td>비유동자산</td>
      <td>-</td>
      <td>제 62 기</td>
      <td>64720596190</td>
      <td>제 61 기</td>
      <td>31496125317</td>
      <td>제 60 기</td>
      <td>34852182924</td>
      <td>1</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_PropertyPlantAndEquipment</td>
      <td>유형자산</td>
      <td>-</td>
      <td>제 62 기</td>
      <td>42434108210</td>
      <td>제 61 기</td>
      <td>30515895159</td>
      <td>제 60 기</td>
      <td>32839273423</td>
      <td>2</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_IntangibleAssetsOtherThanGoodwill</td>
      <td>영업권 이외의 무형자산</td>
      <td>-</td>
      <td>제 62 기</td>
      <td>6348561908</td>
      <td>제 61 기</td>
      <td>256406221</td>
      <td>제 60 기</td>
      <td>363747986</td>
      <td>3</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_InvestmentProperty</td>
      <td>투자부동산</td>
      <td>-</td>
      <td>제 62 기</td>
      <td>6589966350</td>
      <td>제 61 기</td>
      <td></td>
      <td>제 60 기</td>
      <td></td>
      <td>4</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>dart_GoodwillGross</td>
      <td>영업권</td>
      <td>-</td>
      <td>제 62 기</td>
      <td>2712305285</td>
      <td>제 61 기</td>
      <td></td>
      <td>제 60 기</td>
      <td></td>
      <td>5</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>158</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 자본 [member]</td>
      <td>제 62 기</td>
      <td>46236610988</td>
      <td>제 61 기</td>
      <td>23294889189</td>
      <td>제 60 기</td>
      <td>18741261619</td>
      <td>15</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>159</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>연결재무제표  [member]</td>
      <td>제 62 기</td>
      <td>46236614499</td>
      <td>제 61 기</td>
      <td>23294889189</td>
      <td>제 60 기</td>
      <td>18741261619</td>
      <td>15</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>160</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|소수주주지분</td>
      <td>제 62 기</td>
      <td>3511</td>
      <td>제 61 기</td>
      <td></td>
      <td>제 60 기</td>
      <td></td>
      <td>15</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>161</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 자본 [member]|연결자본잉여금</td>
      <td>제 62 기</td>
      <td>41304925854</td>
      <td>제 61 기</td>
      <td>21429147754</td>
      <td>제 60 기</td>
      <td>20765899034</td>
      <td>15</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>162</th>
      <td>20160527000376</td>
      <td>11011</td>
      <td>2015</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 자본 [member]|자본금 [...</td>
      <td>제 62 기</td>
      <td>13791812000</td>
      <td>제 61 기</td>
      <td>11430685000</td>
      <td>제 60 기</td>
      <td>5819523500</td>
      <td>15</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>163 rows × 18 columns</p>
</div>



오픈 다트 API를 통해 삼성제약의 2015년 연간 연결재무제표를 쉽게 불러왔다. 불러온 데이터프레임을 각자의 필요에 맞춰 가공하여 사용하면 유용할 것이다.

2015년 기준으로 데이터를 불러왔을 때, 2015년 이전 2개년도 정보도 함께 확인 가능하다. 2024년 2월 13일 기준으로 2013년 재무제표까지 오픈 다트 API를 통해 얻을 수 있다. 3개년도 정보를 한번에 확인 가능하므로 전체 연도 정보를 불러오고자 할땐 3년 단위 정보를 불러오면 될 것 같다. 현재 2015년 기준으로 60기, 61기, 62기 재무제표를 불러왔으므로 2018년을 기준으로 불러온다면 63기, 64기, 65기 재무제표를 불러올 수 있을 것이다.


```python
df_fs_2018 = get_financial_statements(
    DART_API_KEY=DART_API_KEY,
    CORP_CODE = df_code.loc[df_code['corp_name'] == '삼성제약', 'corp_code'].values[0],
    BSNS_YEAR = '2018',
    REPRT_CODE = '11011',
    FS_DIV = 'CFS',
)

df_fs_2018
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rcept_no</th>
      <th>reprt_code</th>
      <th>bsns_year</th>
      <th>corp_code</th>
      <th>sj_div</th>
      <th>sj_nm</th>
      <th>account_id</th>
      <th>account_nm</th>
      <th>account_detail</th>
      <th>thstrm_nm</th>
      <th>thstrm_amount</th>
      <th>frmtrm_nm</th>
      <th>frmtrm_amount</th>
      <th>bfefrmtrm_nm</th>
      <th>bfefrmtrm_amount</th>
      <th>ord</th>
      <th>currency</th>
      <th>thstrm_add_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_CurrentAssets</td>
      <td>유동자산</td>
      <td>-</td>
      <td>제 65 기</td>
      <td>79724616717</td>
      <td>제 64 기</td>
      <td>60978260818</td>
      <td>제 63 기</td>
      <td>47711452766</td>
      <td>1</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_CashAndCashEquivalents</td>
      <td>현금및현금성자산</td>
      <td>-</td>
      <td>제 65 기</td>
      <td>3151357753</td>
      <td>제 64 기</td>
      <td>2931620456</td>
      <td>제 63 기</td>
      <td>5390180360</td>
      <td>2</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>dart_ShortTermDepositsNotClassifiedAsCashEquiv...</td>
      <td>단기금융자산</td>
      <td>-</td>
      <td>제 65 기</td>
      <td>1200000000</td>
      <td>제 64 기</td>
      <td>1200000000</td>
      <td>제 63 기</td>
      <td>1200000000</td>
      <td>3</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>dart_ShortTermTradeReceivable</td>
      <td>매출채권</td>
      <td>-</td>
      <td>제 65 기</td>
      <td>16699562155</td>
      <td>제 64 기</td>
      <td>15432793585</td>
      <td>제 63 기</td>
      <td>15540472145</td>
      <td>4</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>dart_CurrentFinancialAssetHeldForTrading</td>
      <td>단기매매금융자산</td>
      <td>-</td>
      <td>제 65 기</td>
      <td>3816549555</td>
      <td>제 64 기</td>
      <td>4230007815</td>
      <td>제 63 기</td>
      <td>4052931600</td>
      <td>5</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 지분 [member]</td>
      <td>제 65 기</td>
      <td>131662977152</td>
      <td>제 64 기</td>
      <td>93868490973</td>
      <td>제 63 기</td>
      <td>82377763900</td>
      <td>17</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>173</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|비지배지분 [member]</td>
      <td>제 65 기</td>
      <td></td>
      <td>제 64 기</td>
      <td></td>
      <td>제 63 기</td>
      <td>1013</td>
      <td>17</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>174</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>연결재무제표 [member]</td>
      <td>제 65 기</td>
      <td>131662977152</td>
      <td>제 64 기</td>
      <td>93868490973</td>
      <td>제 63 기</td>
      <td>82377764913</td>
      <td>17</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>175</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 지분 [member]|이익잉여금...</td>
      <td>제 65 기</td>
      <td>-3379583894</td>
      <td>제 64 기</td>
      <td>-49654622640</td>
      <td>제 63 기</td>
      <td>-43434867861</td>
      <td>17</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>176</th>
      <td>20190401003716</td>
      <td>11011</td>
      <td>2018</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 지분 [member]|기타포괄손...</td>
      <td>제 65 기</td>
      <td>24610931430</td>
      <td>제 64 기</td>
      <td>24646132247</td>
      <td>제 63 기</td>
      <td>12765714568</td>
      <td>17</td>
      <td>KRW</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 18 columns</p>
</div>



다음과 같이 'REPRT_CODE' 인자를 수정해 3분기 데이터를 불러올 수 있다. 1분기, 반기 데이터도 해당 요청 인자를 수정하여 확인할 수 있다.


```python
df_fs_3q = get_financial_statements(
    DART_API_KEY=DART_API_KEY,
    CORP_CODE = df_code.loc[df_code['corp_name'] == '삼성제약', 'corp_code'].values[0],
    BSNS_YEAR = '2016',
    REPRT_CODE = '11014',
    FS_DIV = 'CFS',
)

df_fs_3q
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rcept_no</th>
      <th>reprt_code</th>
      <th>bsns_year</th>
      <th>corp_code</th>
      <th>sj_div</th>
      <th>sj_nm</th>
      <th>account_id</th>
      <th>account_nm</th>
      <th>account_detail</th>
      <th>thstrm_nm</th>
      <th>...</th>
      <th>frmtrm_nm</th>
      <th>frmtrm_amount</th>
      <th>bfefrmtrm_nm</th>
      <th>bfefrmtrm_amount</th>
      <th>ord</th>
      <th>currency</th>
      <th>thstrm_add_amount</th>
      <th>frmtrm_q_nm</th>
      <th>frmtrm_q_amount</th>
      <th>frmtrm_add_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_NoncurrentAssets</td>
      <td>비유동자산</td>
      <td>-</td>
      <td>제 63 기 3분기말</td>
      <td>...</td>
      <td>제 62 기말</td>
      <td>64720596190</td>
      <td>제 61 기말</td>
      <td>31496125317</td>
      <td>1</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_PropertyPlantAndEquipment</td>
      <td>유형자산</td>
      <td>-</td>
      <td>제 63 기 3분기말</td>
      <td>...</td>
      <td>제 62 기말</td>
      <td>42434108210</td>
      <td>제 61 기말</td>
      <td>30515895159</td>
      <td>16</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_IntangibleAssetsOtherThanGoodwill</td>
      <td>영업권 이외의 무형자산</td>
      <td>-</td>
      <td>제 63 기 3분기말</td>
      <td>...</td>
      <td>제 62 기말</td>
      <td>6348561908</td>
      <td>제 61 기말</td>
      <td>256406221</td>
      <td>56</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>dart_GoodwillGross</td>
      <td>영업권</td>
      <td>-</td>
      <td>제 63 기 3분기말</td>
      <td>...</td>
      <td>제 62 기말</td>
      <td>2712305285</td>
      <td>제 61 기말</td>
      <td></td>
      <td>75</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>BS</td>
      <td>재무상태표</td>
      <td>ifrs_OtherNoncurrentFinancialAssets</td>
      <td>기타비유동금융자산</td>
      <td>-</td>
      <td>제 63 기 3분기말</td>
      <td>...</td>
      <td>제 62 기말</td>
      <td>6635654437</td>
      <td>제 61 기말</td>
      <td>723823937</td>
      <td>85</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>174</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 자본 [member]</td>
      <td>제 63 기 3분기</td>
      <td>...</td>
      <td>제 62 기</td>
      <td>46236610988</td>
      <td>제 61 기</td>
      <td>23294889189</td>
      <td>24</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>제 62 기 3분기</td>
      <td>36795377012</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>175</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>연결재무제표  [member]</td>
      <td>제 63 기 3분기</td>
      <td>...</td>
      <td>제 62 기</td>
      <td>46236614499</td>
      <td>제 61 기</td>
      <td>23294889189</td>
      <td>24</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>제 62 기 3분기</td>
      <td>36795381071</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>176</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|비지배지분 [member]</td>
      <td>제 63 기 3분기</td>
      <td>...</td>
      <td>제 62 기</td>
      <td>3511</td>
      <td>제 61 기</td>
      <td></td>
      <td>24</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>제 62 기 3분기</td>
      <td>4059</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>177</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 자본 [member]|미처분이익...</td>
      <td>제 63 기 3분기</td>
      <td>...</td>
      <td>제 62 기</td>
      <td>-21702923038</td>
      <td>제 61 기</td>
      <td>-23748181059</td>
      <td>24</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>제 62 기 3분기</td>
      <td>-22832410583</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>178</th>
      <td>20161129000446</td>
      <td>11014</td>
      <td>2016</td>
      <td>00126414</td>
      <td>SCE</td>
      <td>자본변동표</td>
      <td>ifrs_Equity</td>
      <td>기말자본</td>
      <td>자본 [member]|지배기업의 소유주에게 귀속되는 자본 [member]|자본금 [...</td>
      <td>제 63 기 3분기</td>
      <td>...</td>
      <td>제 62 기</td>
      <td>13791812000</td>
      <td>제 61 기</td>
      <td>11430685000</td>
      <td>24</td>
      <td>KRW</td>
      <td>NaN</td>
      <td>제 62 기 3분기</td>
      <td>13223954500</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>179 rows × 21 columns</p>
</div>


## 마무리

위에서 다룬 것과 유사한 방식으로 오픈 다트 API에서 제공하는 다양한 기업 정보를 불러와서 사용할 수 있다. 근 2주 동안 저평가 분석기를 만들기 위해 플러터를 공부하면서 '앱 프론트앤드 부분은 어떻게 하겠는데 원하는 기능을 구현하기 위해 필요한 데이터를 쉽게 구할 수 있을까' 하는 걱정이 있었다. 오늘 오픈 다트 API를 알아보며 대부분 필요한 정보를 다트를 통해 구할 수 있을 것 같아 조금 마음이 놓인다. 이제는 구체화한 기능을 구현하기 위해 꼭 필요한 데이터를 구분하고, 데이터베이스를 어떻게 설계할지 공부하고, 파이어베이스나 수파베이스 활용법을 알아보며 백엔드 부분을 준비해야겠다.

## 참고자료

|소스|링크|일자|
|:---|:---|:---|
|Youtube|[\[파이썬 퀀트\] 18강 - DART API를 이용해 공시정보 및 재무제표 수집하기](https://www.youtube.com/watch?v=aYmUbU9aR4Q&list=PLkREQFGfPOZ3g1aWNVEHoJRaUWOga10pw&index=11)|230307|