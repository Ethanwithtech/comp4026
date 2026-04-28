tell application "Microsoft Word"
    activate
    set theInput to POSIX file "/Users/yuchendeng/Desktop/comp4026/comp4026/22256342_Written_Assignment.docx" as alias
    set theOutput to "/Users/yuchendeng/Desktop/comp4026/comp4026/22256342_Written_Assignment.pdf"
    open theInput
    set activeDoc to active document
    save as activeDoc file name theOutput file format format PDF
    close activeDoc saving no
end tell
