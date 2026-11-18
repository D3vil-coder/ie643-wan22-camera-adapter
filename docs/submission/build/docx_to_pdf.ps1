# Convert the report .docx to PDF with Microsoft Word (COM). Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File build\docx_to_pdf.ps1 [-Docx path] [-Pdf path]
param(
  [string]$Docx = (Join-Path $PSScriptRoot "..\NeuralReaper_Submission2_Report.docx"),
  [string]$Pdf  = (Join-Path $PSScriptRoot "..\NeuralReaper_Submission2_Report.pdf")
)
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::GetFullPath($Pdf)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
try {
  $doc = $w.Documents.Open($Docx, $false, $true)
  $doc.Fields.Update() | Out-Null
  $doc.ExportAsFixedFormat($Pdf, 17)   # 17 = wdExportFormatPDF
  $pages = $doc.ComputeStatistics(2)   # 2 = wdStatisticPages
  $doc.Close(0)
  Write-Output "pages: $pages"
} finally { $w.Quit() }
