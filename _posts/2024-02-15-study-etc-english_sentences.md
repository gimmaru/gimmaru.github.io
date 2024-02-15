---
layout: post
title:  "[영어] 막힌 문장 분석 (~ 24.02.15)"
subtitle:   "부제"
categories: study
tags: etc
comments: true

---

해석이 막힌 문장들에 대해 문장, 모르는 단어, 구조, 해석 순으로 정리

정리된 문장들은 종종 확인하면서 복습하기

### 240131

---
출처: [코딩셰프 유튜브](https://www.youtube.com/watch?v=AdYRASHRKwE&list=PLQt_pzi-LLfpcRFhWMywTePfZ2aPapvyl&index=1)


Flutter was always architected as a portable UI toolkit and, among other places, runs on Windows, Mac, Fuchsia, and even Raspberry Pi.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

was architected: 설계되다. <br />
among other places: '특히', '무엇보다도' 등의 의미로 해석됨. <br />
[run](https://blog.naver.com/namsam76/222250926353): 움직임을 나타내는 자동사로 쓰이면 1형식

2형식과 1형식으로 이루어진 문장이다.

플러터는 휴대가능한 UI 툴킷으로 설계되었으며, 특히 윈도우, 맥, 퓨시아 그리고 심지어 라즈베리파이에서도 동작한다.
</div>
</details>

---
출처: StackOverflow

The fact that the line actually in your .zshrc file contains this would appear to be an error. In this case it will write the output from the export command to the .zshrc file every time you log in. The export command outputs nothing, so this extra part will essentially do nothing, and should probably be removed so you are left with just the first line.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

- The fact that the line actually in your .zshrc file contains this would appear to be an error.

    S: <u>The fact / that S: ( the line actually in your .zshrc file ) V: contains O: this</u> V: <u>would appear</u> ( to be an error ).

    appear: 1형식 완전 자동사 || 2형식 불완전 자동사<br />
    \* 2형식 불완전 자동사인 경우 보어 O, 목적어 X

    1형식 안에 3형식이 포함되어 있는 문장이다.

    실제로 너의 .zshrc 파일의 라인은 이것을 포함한다는 사실은 에러로 보일 수 있다.

- In this case it will write the output from the export command to the .zshrc file every time you log in.
    
    ( In this case ) S: <u>it</u> V: <u>will write</u> O: <u>the output</u> ( from the export command to the .zshrc file every time / S: you V: log in. )

    이 경우 이것은 너의 export 커맨드 출력을 로그인 할때마다 .zshrc 파일에 적을 것이다.(기록합니다.)

- The export command outputs nothing, so this extra part will essentially do nothing, and should probably be removed so you are left with just the first line.

    output: 명사 - 생산량, 산출량, 출력 \| 동사 - 출력해 내다.

    S: <u>The export command</u> V: <u>outputs</u> O: <u>nothing</u>

    S: <u>The export command</u> V: <u>should probably be removed</u> / so you are left with just the first line.

    be left with: ···를 남기다.

    export 커맨드는 아무것도 출력하지 않는다. 그래서 이 추가적인 부분은 본질적으로 아무것도 안 하고, 아마도 지워져야만 한다. 그러므로 너는 단지 첫번째 라인만 남기면 된다.

    > 내보내기 명령은 아무 것도 출력하지 않으므로 이 추가 부분은 본질적으로 아무 일도 하지 않으므로 첫 번째 줄만 남도록 제거해야 합니다. (deepl)
</div>
</details>

---
출처: Android Studio

hide obsolete packages

\* obsolete: 더 이상 쓸모가 없는, 한물간, 구식의(=out of date)

---
출처: [How to find, open, and edit zshrc file on Mac](https://macpaw.com/how-to/zshrc-file-mac)

Once created, it has a period before its name, meaning it's a hidden file, so technically, its name is '.zshrc'

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

( Once created ), S: <u>it</u> V: <u>has</u> O: <u>a period</u> ( before its name ), ( meaning it's a hidden file ), ( so technically, its name is '.zshrc' )

한 번 생성되면, 이것의 이름 전에 이것은(.zshrc) 주기를 갖는다. 숨겨진 파일임을 의미하며, 기술적으로 이것의 이름은 .zshrc이다.

> 일단 생성되면 이름 앞에 마침표가 붙어 숨겨진 파일임을 의미하므로, 엄밀히 말하면 파일 이름은 '.zshrc'입니다. (deepl)

period: 기간, 시기 \| 시대 \| 이상, 끝(논쟁을 끝내고 더 이상의 말을 할 필요가 없을 때 사용) \| **마침표**<br />
\* **마침표**: (Am) period, (Brit) full stop<br />
technically: 엄말히 따지면\[말하면\] \| 기술적으로\[기법상으로\]
</div>
</details>

---
출처: [How to find, open, and edit zshrc file on Mac](https://macpaw.com/how-to/zshrc-file-mac)

Strip away the GUI in macOS, and what you’re left with is a UNIX core. 

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

strip away: (막 같은 것을, 사실이 아닌 것·불필요한 것을) 벗겨내다<br />
be left with: ···를 남기다.

macOS의 GUI를 벗겨 내고, 너에게 남는 것은 유닉스 코어다.

> macOS에서 GUI를 제거하면 UNIX 코어만 남게 됩니다. (deepl)
</div>
</details>

---
출처: [runApp function](../../development/flutter/packages/flutter/lib/src/widgets/binding.dart)

Inflate the given widget and attach it to the screen.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

inflate: 부풀리다, 과장하다, (가격) 오르다.

위젯을 올리고 화면에 부착합니다.

> 주어진 위젯을 부풀려서 화면에 부착합니다. (deepl)

(Inflate이 '위젯을 어느 공간으로 떠올린다'는 느낌으로 사용되는 것 같다.)
</div>
</details>
<br />

### 240201

---
출처: [코딩셰프]()

A handle to the location of a widget in the widget tree.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

handle: v. 다루다 \| 만지다 \| n. 손잡이

위젯 트리에서 위젯의 위치 손잡이

> 위젯 트리에서 현재 위젯의 위치를 알 수 있는 정보 (코딩셰프)
</div>
</details>

---
출처: [Flutter documentation](https://docs.flutter.dev/tools/formatting)

The alternative is often tiring formatting debates during code reviews, where time might be better spent on code behavior rather than code style.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

tiring: a. 피곤하게 만드는, 피곤한 (=exhausting)<br />
alternative: n. 대안, 선택 가능한 것 \| a. 대체 가능한, 대안이 되는, 대안적인, 대체의<br />
spend on: ···에 (돈을 )쓰다.

❗️ S: <u>The alternative</u> V: <u>is</u> C: <u>often tiring formatting debates</u> ( during code reviews ), ( where S: time V: might be better spent ( on code behavior ) rather than code style ).

대안은 코드 리뷰 중 종종 피곤한 포메팅 관련 토론이다. 이때 코드 스타일보다 코드 작성에 더 많은 시간을 쏟는게 나을지 모른다.

```bash
코드가 선호하는 스타일을 따를 수도 있지만(경험상), 개발자 팀에서는 이렇게 하는 것이 더 생산적일 수 있습니다:

하나의 공유 스타일을 사용하고
자동 서식 지정을 통해 이 스타일을 적용합니다.

그렇지 않으면 코드 검토 중에 지루한 서식 논쟁을 벌이는 경우가 많습니다. 코드 스타일보다는 코드 동작에 더 많은 시간을 할애할 수 있습니다. (deepl)
```
</div>
</details>
<br />

### 240202

---
출처: [Flutter \> Widget \> Builder class](https://api.flutter.dev/flutter/widgets/Builder-class.html)

A stateless utility widget whose build method uses its builder callback to create the widget's child.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

A stateless utility widget whose S: <u>build method</u> V: <u>uses</u> O: <u>its builder callback</u> ( to create the widget's child ).

whose: 한정사, 대명사 (의문문에서) 누구의 \| 한정사, 대명사 <소유의 의미와 함께 어떤 사람·사물을 수식하는 형용사절을 이끄는 데 씀> \| 한정사, 대명사 <소유의 의미와 함께 어떤 사람·사물에 대해 정보를 덧붙일 때 씀><br />
callback: 회신 \| 답신 전화

빌드 메서드가 위젯의 자식을 생성하는데 자신의 빌더 회신(출력)을 사용하는 stateless 위젯

> 빌드 메서드가 빌더 콜백을 사용하여 위젯의 자식을 생성하는 상태 비저장 유틸리티 위젯입니다. (deepl)
</div>
</details>

---

출처: [Flutter \> Widget \> Builder class](https://api.flutter.dev/flutter/widgets/Builder-class.html)

This widget is an inline alternative to defining a StatelessWidget subclass.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

❗️ S: <u>This widget</u> V: <u>is</u> C: <u>an inline alternative</u> ( to defining a StatelessWidget subclass. )

? 이 위젯은 무상태위젯 서브 클래스를 정의하는 한줄 대안이다.

> 이 위젯은 StatelessWidget 서브클래스를 정의하는 대신 사용할 수 있는 인라인 대안입니다. (deepl)

\* alternative to ··· : <u>to 이하를 대신하는</u> 정도로 해석하면 되는 것 같다.<br />
\* inline: 한줄로 해석하는 것보다 라인 내부에서 정도로 생각하고 넘어가는게 맞는 것 같다.

inline: a. 인라인의, 그때마다 즉시 처리하는, 일렬로 늘어선(직렬의)

</div>
</details>

---

출처: [Flutter \> Widget \> Builder class](https://api.flutter.dev/flutter/widgets/Builder-class.html)

The difference between either of the previous examples and creating a child directly without an intervening widget, is the extra BuildContext element that the additional widget adds. This is particularly noticeable when the tree contains an inherited widget that is referred to by a method like Scaffold.of, which visits the child widget's BuildContext ancestors.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

- The difference between either of the previous examples and creating a child directly without an intervening widget, is the extra BuildContext element that the additional widget adds.

    S: <u>The difference ( between either of the previous examples and creating a child directly without an intervening widget )</u>, V: <u>is</u> C: <u>the extra BuildContext element ( that the additional widget adds )</u>.

    \* intervening: a. (두 사건·날짜·사물 등의) 사이에 오는\[있는\]<br />
    intervene: v. (상황 개선을 돕기 위해) 개입하다 \| v. (다른 사람이 말하는 데) 끼어들다\[가로막다] \| v. (방해가 되는 일이) 생기다\[일어나다]

    앞선 예시와 사이에 낀 위젯없이 직접 자식 위젯을 생성하는 것의 차이는 ! <u>새로운 위젯이 추가된 빌드컨텍스트 엘리먼트이다.</u>

    > 앞의 예시와 위젯을 개입시키지 않고 직접 자식을 만드는 것의 차이점은 추가 위젯이 추가하는 BuildContext 요소에 있습니다. (deepl)

    차이는 추가 위젯이 추가하는 Buildcontext 요소에 있다. → '만들어낸 위젯이 새로운 컨텍스트를 추가하는지' deepl 해석이 맞다.<br />
    \* "that the additional widget adds"를 "추가된 위젯이 추가하는" 같이 형용사처럼 해석해야한다.

- This is particularly noticeable when the tree contains an inherited widget that is referred to by a method like Scaffold.of, which visits the child widget's BuildContext ancestors.

    S: <u>This</u> V: <u>is</u> ( particularly ) C: <u>noticeable</u> ( when S: the tree V: contains O: an inherited widget ( that V: is C: referred to ( by a method like Scaffold.of ), ( S: which V: visits O: the child widget's BuildContext ancestors. ) ) ) 

    ancestor: n. 조상, 선조 \| n. (기계의) 원형

    이것은 위젯 트리가 자식 위젯의 빌드컨텍스트 조상을 방문하는 'Scaffold.of' 같은 메서드로 참조되는 내부적인 위젯을 포함하고 있을 때 특히 주목할만하다.

    > 이는 트리에 상속된 위젯이 포함되어 있을 때 특히 두드러지는데, 이 위젯은 자식 위젯의 BuildContext 조상을 방문하는 Scaffold.of와 같은 메서드에 의해 참조됩니다. (deepl)

</div>
</details>
<br />

### 240205

---
출처: [3Blue1Brown Survey](https://docs.google.com/forms/d/e/1FAIpQLSezVpJ3CPjzjjmZ1ICP4JmrX4PUHahCvfp6DLZR2-zARuFdiQ/viewform)

By filling out this form, we may reach out to you with examples of AI-generated dubbings on 3b1b videos, and ask for feedback on how natural they sound. We may also invite you to provide feedback/edits on the underlying text translation.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

- By filling out this form, we may reach out to you with examples of AI-generated dubbings on 3b1b videos, and ask for feedback on how natural they sound.

    \* reach out: 1. 뻗다 \| 자라다 \| ···와 접촉하려고 하다 \| (대중에게)연락을 취하려하다.

    ( By filling out this form ), S: <u>we</u> V: <u>may reach out</u> ( to you with examples of AI-generated dubbings on 3b1b videos ), and V: <u>ask</u> ( for feedback on how natural they sound ).

    이 서식을 채우면, 우리는 너에게 3b1b 비디오 AI 생성 더빙 사례와 함께 연락을 취할 것이고, 비디오들이 얼마나 자연스럽게 들리는지 물어보려고 한다.

    > 이 양식을 작성하시면 3b1b 동영상에서 AI가 생성한 더빙의 예시를 보여드리고, 얼마나 자연스러운지 피드백을 요청할 수 있습니다. (deepl)

- We may also invite you to provide feedback/edits on the underlying text translation.

    underlying: (겉으로 잘 드러나지는 않지만) 근본적인\[근원적인\] \| (다른 것의) 밑에 있는 (→underlie) \| ···의 기저를 이루다

    S: <u>We</u> V: <u>may also invite</u> O: <u>you</u> O.C: <u>to provide feedback/edits</u> ( on the underlying text translation ).

    또한, 우리는 비디오에 깔린 자막 번역에 대한 피드백과 편집을 받기 위해 당신을 초대할 수도 있다.

    > 또한 기본 텍스트 번역에 대한 피드백이나 수정 사항을 제공하도록 요청할 수도 있습니다. (deepl)

</div>
</details>
<br />

### 240208

---
출처: [maybe-finance/maybe](https://github.com/maybe-finance/maybe?tab=readme-ov-file)

We spent the better part of $1,000,000 building the app.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

\* better part: 과반수, 대부분

S: <u>We</u> V: <u>spent</u> O: <u>the better part</u> ( of $1,000,000 ) ( building the app ).

우리는 백만 달러의 대부분을 앱 구축에 썼다.

> 앱 구축에 1,000,000달러 이상을 투자했습니다. (deepl)

</div>
</details>

---
출처: [XAMPP Apache + MariaDB + PHP + Perl](http://gimmarus-macbook-air.local/dashboard/)

XAMPP is meant only for development purposes.

---
출처: [XAMPP Apache + MariaDB + PHP + Perl](http://gimmarus-macbook-air.local/dashboard/)

XAMPP is configured to be open as possible to allow the developer anything he/she wants.

### 240214

---
출처: [Updates to our Terms of Use](https://mail.naver.com/v2/read/-1/12287)

You are receiving this email because you have been identified in our system as being an “Administrator” or “Owner” (“Admin/Owner”) of a registered Calendly Account(s) subject to our online Terms of Use (“Customer Terms”).

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

\* subject to: 1. 부사 \[법률\] ~(의 조건)에 따라, ~의 조건하에, ~를 조건으로 하여 \| 2. \[형용사\] ~을 조건으로 하는 , ~을 받아야 하는 \| \[형용사\] ~의 영향하에 있는
<br />

무언가에 종속된다는 개념인 것 같다.

subject: \[명사\] 주제, 대상, 주어, 과제, 학과, 과목

S: <u>You</u> V: <u>are</u> C: <u>receiving ( this email ) </u> ( because you have been identified ) ( in our system ) ( as being an “Administrator” or “Owner” (“Admin/Owner”) )  ( of a registered Calendly Account(s) subject to our online Terms of Use (“Customer Terms”) ).

우리 시스템에서 고객 용어에 영향을 받는 캘린들리 계정의 관리자 혹은 소유권자로 식별되었기 때문에 이 이메일을 받으셨습니다.

> 귀하가 당사의 온라인 이용약관("고객 약관")에 따라 등록된 캘린더 계정의 "관리자" 또는 "소유자"("관리자/소유자")로 시스템에서 확인되었으므로 이 이메일을 수신하는 것입니다.

\* terms of use: 이용 약관

</div>
</details>

---
출처: [supabase DOCS - Getting Started - Features](https://supabase.com/docs/guides/getting-started/features)

This is a non-exhaustive list of features that Supabase provides for every project.

<details>
<summary><b>분석</b></summary>
<div markdown="1">
<br />

\* non-exhaustive: 불완전한 (↔ exhaustive: (하나도 빠뜨리는 것 없이) 철저한\[완전한\])

S: <u>This</u> V: <u>is</u> C: <u>a non-exhaustive list</u> ( of features that Supabase provides for every project ).

수파베이스가 모든 프로젝트에 제공하는 불완전한 기능 목록이다.

> 이것은 모든 프로젝트에 대해 Supabase가 제공하는 기능의 전체 목록이 아닙니다. (deepl)

딥엘 해석을 보니 불완전한말고 '전체 목록이 아니다'라고 해석하는게 더 맞는 것 같다.

</div>
</details>

<br />

## 참고자료

|소스|링크|일자|
|:---|:---|:---|
|네이버 블로그|[영어 2형식 문장, 2형식 기본문장, 2형식 단어, be 동사(연결동사)](https://blog.naver.com/namsam76/222269590234)|210309|
|티스토리|[\[영문법\]2. 현재진행형(be+~ing) 뜻과 활용 쉬운 정리](https://dancing-english.tistory.com/12)|200511|
|티스토리|[VS Code에서 나만의 Snippets 만들기](https://jojoldu.tistory.com/491)|200404|
|Nesoy Blog|[Vscode Code Snippets 설정하기](https://nesoy.github.io/articles/2019-03/Vscode-Code-snippet)|190303|
|티스토리|[⚡ [꿀팁모음/VSC] Visual Studio Code 단축키, 여러 줄 편집하기 등 팁 모음](https://y-oni.tistory.com/57)|210324|
