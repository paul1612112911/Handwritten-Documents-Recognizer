label_ids = {
    "ManualInputDialogTitle",
    "ManualInputDialogHint",
    "LoadTemplateBtn",
    "ClearBtn",
    "RecogFileDocBtn",
    "RecogFolderDocBtn",
    "EndEditAndSaveBtn",
    "LoadTemplateLabel",
    "SelectRecogAreaLabel",
    "FinishSelectLabel",
    "SelectingLabel",
    "RecogDoneLabel",
    "SaveResultWindowTitle",
    "LoadingTemplateBtn",
}

ChineseDict = {
    "ManualInputDialogTitle": '編輯欄位',
    "ManualInputDialogHint": '編輯欄位',
    "LoadTemplateBtn": "載入模板",
    "ClearBtn": '清除',
    "RecogFileDocBtn": '辨識圖片檔案',
    "RecogFolderDocBtn": '辨識圖片資料夾',
    "EndEditAndSaveBtn": '結束編輯並儲存',
    "LoadTemplateLabel": '請載入模板',
    "SelectRecogAreaLabel": '請框選辨識目標',
    "FinishSelectLabel": '可啟動辨識或框選其他目標',
    "SelectingLabel": '請選擇第二個點',
    "RecogDoneLabel": '辨識完成，點選欄位進行編輯，滑鼠滾輪切換文件。',
    "SaveResultWindowTitle": '儲存辨識結果',
    "LoadingTemplateBtn": '正在載入模板',
}
EnglishDict = {
    "ManualInputDialogTitle": "Edit Field",
    "ManualInputDialogHint": "Edit Field",
    "LoadTemplateBtn": "Load Template",
    "ClearBtn": "Clear",
    "RecogFileDocBtn": "Recognize Image File",
    "RecogFolderDocBtn": "Recognize Image Folder",
    "EndEditAndSaveBtn": "Finish Editing and Save",
    "LoadTemplateLabel": "Please load the template",
    "SelectRecogAreaLabel": "Please select the target area",
    "FinishSelectLabel": "Recognition ready or select another target",
    "SelectingLabel": "Please select the second point",
    "RecogDoneLabel": "Recognition complete. Click a field to edit. Use mouse wheel to switch files.",
    "SaveResultWindowTitle": "Save Recognition Result",
    "LoadingTemplateBtn": "Loading template",
}


languages = {
    "Chinese": ChineseDict,
    "English": EnglishDict,
}


class TextGetter:
    def __init__(self):
        self.languages = languages
        self.display_language = "Chinese"
        self.ids = label_ids
        for language_name, language_dict in self.languages.items():
            lang_set = set(language_dict.keys())
            too_many = lang_set - self.ids
            too_few = self.ids - lang_set
            assert lang_set == self.ids, (
                f"the language '{language_name}' has "
                f"{'too many names: ' + str(too_many) if too_many else ''} "
                f"{'too few names: ' + str(too_few) if too_few else ''}"
            )

    def set_language(self, language: str):
        assert language in self.languages, f"this language '{language}' is not in dictionaries: {self.languages.keys()}."
        self.display_language = language

    def get_text(self, text_id: str):
        assert text_id in self.ids, f"this id '{text_id}' is not in dictionaries."
        return self.languages[self.display_language][text_id]

if __name__ == "__main__":
    tg = TextGetter()
    tg.set_language("Chinese")
    print(tg.get_text("ClearBtn"))  # Output: Clear

