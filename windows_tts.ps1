param(
  [Parameter(Mandatory=$true)][string]$Text,
  [Parameter(Mandatory=$true)][string]$Out,
  [Parameter(Mandatory=$true)][int]$Rate,
  [Parameter(Mandatory=$false)][string]$VoiceHint = "female"
)

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo } | Where-Object { $_.Culture.Name -like "zh-*" })
if ($voices.Count -eq 0) {
  throw "没有找到中文系统语音。请在 Windows 设置中安装中文语音包。"
}
$selected = $voices | Where-Object { $_.Gender.ToString().ToLower() -eq $VoiceHint } | Select-Object -First 1
if ($null -eq $selected) { $selected = $voices[0] }
$synth.SelectVoice($selected.Name)
$synth.Rate = [Math]::Max(-10, [Math]::Min(10, [int](($Rate - 180) / 10)))
$synth.SetOutputToWaveFile($Out)
$synth.Speak($Text)
$synth.Dispose()
