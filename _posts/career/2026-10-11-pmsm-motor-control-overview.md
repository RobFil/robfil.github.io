---
layout: post
title: "PMSMモーター制御：センサーから電圧出力まで"
date: 2026-10-11
categories: [career]
topic: "Bewerbungsgespräch / モーター制御"
summary_de: "Verständlicher Überblick über die PMSM-Regelkette: Sensorik, feldorientierte Stromregelung, PWM-Spannungsausgabe, Ziele und technische Herausforderungen."
generated: false
source_urls:
  - "https://www.ti.com/lit/an/sprabq2/sprabq2.pdf"
  - "https://www.ti.com/document-viewer/lit/html/SBOA587/GUID-915C0393-540E-4DBA-B00B-2A6B52D87277"
  - "https://community.st.com/ysqtg83639/attachments/ysqtg83639/power-management-forum/6852/1/STM32F103xx%20PMSM%20FOC%20software%20library%20UM.pdf"
source_checked_at: 2026-10-11
---

## 目的

PMSM（永久磁石同期モーター）の制御の目的は、ドライバーまたは車両の上位機能が要求するトルクを、速く、正確に、効率よく発生させることです。電気自動車では、加速時だけでなく、回生時、低速時、高速時、低温時や高温時にも、滑らかで安全な応答が必要です。したがって、単にモーターを回すだけではなく、トルク、損失、電流、電圧、温度、安全性を同時に管理します。

## 制御信号の流れ

基本的な流れは、次のように説明できます。

`トルク要求 → 電流目標値 → 電流測定・座標変換 → 電流制御 → 電圧指令 → PWM → インバーター → PMSM`

まず、上位機能からトルク要求が届きます。モーター制御ソフトウェアは、その要求をd軸・q軸の電流目標値に変換します。q軸電流は主にトルクに関係します。d軸電流は磁束に関係し、IPMSMでは運転点に応じてMTPA制御に使われます。高速領域では、インバーターの電圧制限の中で回転数を上げるために、負のd軸電流による磁束弱め制御を行うことがあります。

次に、センサーから情報を取得します。代表的な信号は、三相電流、DCリンク電圧、インバーターやモーターの温度、そしてローターの位置と回転数です。ローター位置は、レゾルバやエンコーダで直接測定できます。また、センサーを使わず、電圧・電流とモーターのモデルから推定するセンサーレス制御もあります。位置情報は、どの方向に磁束があるかを知るために必要です。

三相電流はADCでサンプリングされます。測定した電流は、Clarke変換とPark変換によって、回転するd-q座標系の電流に変換されます。この座標系では、磁束に関係する成分とトルクに関係する成分を分けて考えられます。測定値と目標値の差をそれぞれPIコントローラーに入力し、必要なd軸・q軸電圧指令を計算します。

その後、逆Park変換によって電圧指令を三相系に戻し、SVPWMなどの変調方式で各相のデューティ比に変換します。マイクロコントローラーはPWM信号を出力し、ゲートドライバーがインバーターの六つの半導体スイッチを駆動します。インバーターはバッテリーのDC電圧を三相のPWM電圧に変換し、PMSMに必要な電流を流します。この制御ループはPWM周期に同期して繰り返されます。

## 主な課題

第一に、測定精度とタイミングです。電流センサーのオフセット、ノイズ、ADCの量子化誤差、ローター角度の誤差は、トルクの誤差や振動につながります。特にPWMのスイッチング直後は電流信号が乱れやすいため、適切なサンプリング時刻と補正が重要です。

第二に、インバーターとモーターの実際の挙動です。デッドタイム、電圧降下、磁気飽和、温度による抵抗や磁石特性の変化、部品ばらつきは、理想的なモデルとのずれを生みます。そのため、制御機能だけでなく、キャリブレーション、補償、診断も必要です。

第三に、複数の目標の両立です。高い応答性には高い制御帯域が必要ですが、スイッチング周波数を上げると損失、発熱、EMC、計算負荷が増えます。さらに、故障時には過電流、過電圧、温度異常、センサー異常を検出し、安全な状態へ移行しなければなりません。

このように、PMSM制御は「センサーから得た状態をもとに、必要な電流を計算し、インバーターの電圧を高速に調整する」仕組みです。私の仕事では、この制御の機能設計、評価、そして他チームとのインターフェースを技術面とプロジェクト面の両方から考えることが重要です。

## 参考資料

- [Texas Instruments: Sensored Field Oriented Control of 3-Phase PMSM](https://www.ti.com/lit/an/sprabq2/sprabq2.pdf)
- [Texas Instruments: High Speed Single Shunt Current Sensing in Motor Control](https://www.ti.com/document-viewer/lit/html/SBOA587/GUID-915C0393-540E-4DBA-B00B-2A6B52D87277)
- [STMicroelectronics: Introduction to Sensorless FOC of PM Motors](https://community.st.com/ysqtg83639/attachments/ysqtg83639/power-management-forum/6852/1/STM32F103xx%20PMSM%20FOC%20software%20library%20UM.pdf)
