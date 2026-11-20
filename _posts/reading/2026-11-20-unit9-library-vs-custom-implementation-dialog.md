---
layout: reading
title: "Unit 9：既存ライブラリか新規実装かを決める"
date: 2026-11-20
categories: [reading]
topic: Business
level: N2
reading_time_minutes: 10
grammar: ["～さんは、～についてどのようにお考えですか", "少し話を戻して、～について確認してもよろしいでしょうか", "ここまでの議論をまとめると、～", "～という方針でよろしいでしょうか"]
business_expressions: ["～さんは、～についてどのようにお考えですか", "少し話を戻して、～について確認してもよろしいでしょうか", "ここまでの議論をまとめると、～", "～という方針でよろしいでしょうか"]
source_selection: guided
learning_sources: ["20261115_Unit9_会議のファシリテーション_教材.pdf"]
learning_source_details:
  - source_file: "20261115_Unit9_会議のファシリテーション_教材.pdf"
    pages: [1, 6, 10, 11]
    topic_group: "Unit9 会議のファシリテーション"
    document_date: 2026-11-15
summary_de: "Ein Team vergleicht für eine neue Funktion eine vorhandene Bibliothek mit einer Eigenentwicklung. Die Moderation lenkt eine Abschweifung zurück und führt die Teilnehmenden zu einem gemeinsamen Beschluss."
summary_en: "A team compares an existing library with a custom implementation for a new feature. The facilitator redirects an unrelated tangent and guides the group toward a shared decision."
generated: true
fictional: true
source_url: null
source_published_at: null
source_checked_at: null
---

進行役：それでは、実装方法を決めるための会議を始めます。新機能に既存のライブラリを使う案と、必要な部分を新しく実装する案があります。今日は、それぞれの利点と注意点を整理し、どちらで進めるか合意したいと思います。まず、既存のライブラリを調べた担当者から説明してください。

開発担当：既存のライブラリを使えば、開発費用を抑えられます。基本的な機能もすでにあるため、実装を早く始められます。ただ、将来、別の機能を追加したいときに、どこまで対応できるかはまだ分かりません。

進行役：ありがとうございます。新規実装を検討した担当者さんは、拡張性についてどのようにお考えですか。

設計担当：新しく作れば、今回の仕様に合わせて構成できます。将来の機能追加にも対応しやすくなると思います。一方で、設計からテストまで必要なので、工数はかなり増えます。来月のリリースに間に合わせるには、範囲を絞る必要がありそうです。

開発担当：ライブラリなら、まず標準機能でリリースして、利用状況を見てから追加を考えることもできます。最初から全部作るより、リスクが小さいと思います。

設計担当：そうですね。ただ、ライブラリの更新が止まった場合や、必要な機能が追加されない場合には、後から置き換える作業が発生するかもしれません。

進行役：ここまでの議論をまとめると、ライブラリ案は費用と初期の工数を抑えられますが、将来の拡張に不確実さがあります。新規実装案は拡張しやすい一方、工数が増え、日程への影響も考える必要があります。ほかに確認しておきたい点はありますか。

開発担当：画面のボタンは丸い形のほうが使いやすいと思います。色も青にすれば、全体のデザインに合いそうです。

進行役：少し話を戻して、実装方法について確認してもよろしいでしょうか。画面の形や色は、実装方針を決めた後にデザイン担当と相談しましょう。今は、初回リリースで必要な機能をどこまで含めるかを決めたいと思います。

設計担当：それなら、まずライブラリの拡張性を今週中に調査し、必要な機能が追加できるか確認してはどうでしょうか。対応できない機能だけ、新しく作る方法もあります。

開発担当：その進め方なら、費用を抑えながら、将来の変更にも備えられます。調査結果を見て、追加実装の範囲を決めたいです。

進行役：では、今週はライブラリの拡張性を確認し、来週の会議で不足機能と工数を比べてから最終決定する、という方針でよろしいでしょうか。

開発担当・設計担当：はい、それでお願いします。

進行役：ありがとうございます。開発担当さんが調査を進め、設計担当さんは追加実装が必要な場合の見積もりを準備してください。次回はその結果をもとに、リリース日程も含めて決めましょう。

**Summary:** A project team weighs a low-cost existing library against a more extensible but labor-intensive custom implementation, then agrees on an investigation before deciding.

**Grammar and expressions:** ～さんは、～についてどのようにお考えですか (invite a participant’s view); 少し話を戻して、～について確認してもよろしいでしょうか (politely return to the agenda); ここまでの議論をまとめると、～ (organize the discussion); ～という方針でよろしいでしょうか (confirm agreement). These Unit 9 expressions are deliberately revisited to enact role-play 2.
