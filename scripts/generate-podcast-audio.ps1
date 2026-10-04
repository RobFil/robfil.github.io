param(
    [string]$OutputPath = "assets/podcasts/episode-001-gozen-reiji-no-tsuuchi.wav"
)

$ErrorActionPreference = 'Stop'

$text = @'
午前零時の通知。

金曜日の夜、ナナは会社に一人で残っていた。来週公開する予定のアプリを、最後にもう一度確認するためだった。画面を閉じようとしたとき、スマートフォンが小さく震えた。時刻は午前零時三分。「今日も、おつかれさまです」。アプリからの通知だった。

ナナは首をかしげた。その機能は、まだテスト用の環境にしか入っていない。しかも、通知を送る時間は午前九時のはずだった。翌晩も、その次の晩も、同じ時刻に同じ通知が来た。文の最後には、なぜか小さな月の絵文字が付いていた。

月曜日の朝、ナナは同僚のユウタに相談した。「通知の設定ミスが原因で、夜中にメッセージが発生しているのかもしれません」。ユウタはログを開き、「でも、送信元は今のシステムではありません。古いサーバーから来ているようです」と言った。原因は、三年前の試作版がまだ動いていることだと考えられた。

二人は地下の機械室にある古いパソコンを探した。ほこりをかぶった画面には、小さなプログラムが残っていた。開発したのは、すでに別の会社にいる先輩だった。プログラムの説明には、こう書かれていた。「遅くまで働く人が、少しだけ笑えるように」。

ナナはしばらく黙っていた。消してしまえば問題は解決する。しかし、毎晩の通知を待っている人がいるかもしれない。二人は通知をすぐに止めず、利用者への影響範囲については調査中です、とチームに報告した。

その夜、ナナは会社を出る前にアプリを開いた。午前零時三分、画面に月が現れた。「今日も、おつかれさまです」。ナナは笑って、返信のボタンがない画面に小さく言った。「あなたも、おつかれさま」。
'@

$directory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $directory | Out-Null

$speaker = New-Object -ComObject SAPI.SpVoice
$voice = $speaker.GetVoices() |
    Where-Object { $_.GetDescription() -like 'Microsoft Haruka Desktop*' } |
    Select-Object -First 1
if ($null -eq $voice) {
    throw 'Die japanische Windows-Stimme Microsoft Haruka Desktop ist nicht verfügbar.'
}

$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open((Join-Path $PWD $OutputPath), 3, $false)
$speaker.Voice = $voice
$speaker.Rate = -2
$speaker.AudioOutputStream = $stream
$speaker.Speak($text) | Out-Null
$stream.Close()
