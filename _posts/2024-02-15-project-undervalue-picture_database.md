---
layout: post
title:  "[저평가 분석] 그림으로 배우는 데이터베이스"
subtitle:   "부제"
categories: project
tags: undervalue
comments: true

---

백엔드를 어떻게 구성해야할지 찾아보니 데이터베이스가 가장 중요하다고 느꼈다. 그래서 데이터베이스의 기초 개념을 명확히 하고, 설계하는 방법을 알아보고자 관련 책을 찾아보았다.

## 요약

### 1장 데이터베이스의 기본

데이터와 데이터베이스가 무엇인지, 실생활의 사례는 어떤 것이 있는지 말한다. 

데이터베이스는 데이터를 등록, 정리, 검색하는 기본 기능을 갖추고 있고, 데이터베이스 관리 시스템(DBMS)는 이러한 기본 기능에 더해 검색, 제한, 제어, 접근권한, 복구 등의 기능을 추가로 제공해주는 시스템이다. 대용량 데이터를 관리하기 위해 필요한 복잡한 기능들을 제공해준다. 우리가 흔히 말하는 Oracle, MySQL, PostgreSQL, MongoDB 등은 모두 DBMS이고, 데이터베이스를 다루기 위해서 DBMS를 사용한다.

SQL은 DBMS와 소통하기 위해 일반적으로 사용하는 언어다.

### 2장 데이터의 보존 형식

여러 데이터 모델과 그 중 관계형 데이터 모델 가장 일반적으로 사용되는 관계형 모델의 특징을 설명한다.

관계형 데이터베이스의 장점은 데이터의 포맷 통일, 갱신비용 최소화, 정확한 데이터 취득 등이 있고, 

단점은 데이터가 방대해질 때 처리 속도가 늦어진다는 점과, 데이터를 분산할 수 없거나 표현하기 어려운 데이터가 있다는 점이다.

관계형 이 외의 데이터베이스 형식인 NoSQL엔 어떤 것들이 있는지도 확인한다. NoSQL은 관계형 데이터베이스의 단점을 보완하기 위한 목적으로 등장했으며 대용량 데이터를 고속으로 처리하거나 실시간 처리가 요구될 때 종종 사용된다고 한다.

### 3장 데이터베이스 조작

DBMS와 소통하기 위해 필요한 SQL 기본 문법을 소개한다.

### 4장 데이터 관리

데이터베이스에서 활용하는 다양한 데이터 타입과 트랜잭션, 커밋, 롤백이 무엇인지 설명한다.

컬럼별 데이터 타입을 지정하여 지정된 타입만 저장될 수 있게끔 제약하거나, 속성을 붙여 일정한 규칙을 따르게 끔 데이터를 저장하여 데이터를 편하게 관리할 수 있다.

데이터베이스를 대상으로 한 여러가지 처리를 하나로 모은 것을 트랜잭션이라고 한다. 트랜잭션은 원자성, 일관성, 독립성, 영속성 (ACID) 이라는 특성이 있고, 트랜잭션을 끝나고 커밋이 완료되어야 데이터베이스에 반영된다. 만약 트랜잭션 처리 도중 문제가 생기면 롤백하여 처리 이전의 상태로 되돌려 무결성을 유지한다.

복수의 트랜잭션 처리가 동시에 같은 데이터를 조작할 때 데드락이 발생할 수 있다. 이러한 경우를 방지하기 위해 데이터 액세스 순서를 통일하는 등의 대책이 필요하다.

### 5장 데이터베이스 도입

데이터베이스를 도입할 때 고민해야할 부분을 설명한다.

데이터베이스 도입은 크게 요건정의, 설계, 개발, 운용의 순서로 진행된다.

앱을 개발하는 기능과 요구사항에 맞춰 데이터베이스에 저장해 두지 않으면 안되는 정보를 망라해두고 이후에 진행되는 순서인 테이블 설계에 활용한다.

ER 다이어그램을 설계에 활용하면 좋다. 테이블 설계에 있어 필요한 부분을 빠뜨리지 않을 수 있고, 전체 모습을 그릴 수 있어 테이블 설계나 데이터베이스상의 문제점을 특정하는데 도움이 된다.

그 후 정규화가 필요한 부분은 정규화하고 구체화된 컬럼의 데이터 타입, 제약, 속성을 결정한다. 테이블과 컬럼의 이름 명명 규칙에 대해서도 다룬다.

마지막으로 하나의 사례를 통해 데이터베이스를 설계하는 과정을 보여준다.

### 6장 데이터베이스 운용

데이터베이스를 안전하게 운용하기 위해 알아두어야 할 사항을 정리한다.

온프레미스와 클라우드의 차이, DB 서버 관리 시 유의할 점, DB 운용에 드는 비용, 보안을 위해 알아두어야 할 사항, 감시, 백업, 버전 업 등 내용을 다룬다.

### 7장 데이터베이스를 지키기 위한 지식

트러블과 보안을 위한 대책을 설명한다.

물리적 위협, 기술적 위협, 인적 위협과 같이 데이터베이스에 위협이 될 수 있는 요인을 말한다.

문제를 식별하고 해결할 때 도움이 되는 로그 관리, 에러 처리 방법을 소개한다.

DB 시스템을 효과적으로 운영할 수 있도록 슬로우 쿼리, 인덱스, 스케일 업, 스케일 아웃, 리플리케이션 관련 내용도 포함된다.

### 8장 데이터베이스 활용

클라이언트 소프트웨어, WordPress 같은 애플리케이션, 프로그래밍 언어, 클라우드 서비스를 통해 데이터베이스를 활용하는 법을 소개한다.

빅데이터와 인공지능 관련 설명도 있다.

---

## 궁금한 부분들

1. 관계형 데이터베이스 분산 처리가 어려운 이유?

2. 관계형 데이터베이스를 분산 처리하여 사용한 사례 찾아보기

## 느낀 점

데이터베이스 관련 기초 지식을 정리할 수 있어서 좋았다. 하지만 정말 기본적인 내용을 다룬다는 느낌을 받아서 데이터베이스를 잘 구성하기 위해서는 추가적인 공부가 필요할 것 같다.

5장 내용이 특히 좋았다. 큰 틀에서 데이터베이스가 어떤 과정을 거쳐 설계되는지 알 수 있었고, 하나의 예시를 기준으로 직접 설계하는 과정을 보여주는 점이 도움이 됐다.

## 참고할 자료

|소스|링크|일자|
|:---|:---|:---|
|azderica의 블로그|[\[Flink\] Flink이란?](https://azderica.github.io/00-flink/)|201211|
|티스토리|[배워봅시다 Draw.io](https://sjquant.tistory.com/61)|210422|