from typing import Union

class GoogleTransInstallationError(Exception):
    pass

class GoogleTrans:
    
    def __init__(self, src:str = None, dest:str = "en", *args, **kwargs):
        """
        Translate from src language to desc language
        src: source language, if not specified, googletrans will detect the language
        dest: destination language, if not specified, googletrans will translate to English
        *args, **kwargs: parameters for googletrans.Translator
        
        to access the supported languages, use GoogleTrans.languages
        to access the supported language codes, use GoogleTrans.langcodes
        to access the supported language codes and languages, use GoogleTrans._lang_str
        """
        try:
            from googletrans import Translator
        except Exception as e:
            GoogleTransInstallationError("Google translator import error" + e)
        
        if isinstance(src, str) and (not src in self._lang_str):
            raise ValueError(f"Invalid language setting - src: {src}")
        
        if isinstance(dest, str) and (not dest in self._lang_str):
            raise ValueError(f"Invalid language setting - dest: {dest}")
        
        self.src = src
        self.desc = dest
        
        self.translator = Translator(*args, **kwargs)
    
    def translate(self, text:Union[str, list]):
        """tranlsate string text or list of string text"""
        params = {"src":self.src, "dest":self.desc}
        params = {k:v for k,v in params.items() if v is not None}
        result = self.translator.translate(text = text, **params)
        
        if isinstance(result, list):
            return [r.text for r in result]
        
        return result.text
    
    def __call__(self, text:Union[str, list]):
        return self.translate(text)
    
    @property
    def languages(self):
        from googletrans import LANGUAGES
        return list(LANGUAGES.values())
    
    @property
    def langcodes(self):
        from googletrans import LANGUAGES
        return list(LANGUAGES.keys())
    
    
    @property
    def _lang_str(self):
        return self.langcodes + self.languages