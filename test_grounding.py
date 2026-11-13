"""Live regression of the observed unsupported intake claim; no canned LLM replies."""
import unittest, re
from main import Assistant

class GroundingTest(unittest.TestCase):
    def test_missing_answer_does_not_invent_source_identifiers(self):
        r=Assistant().ask('Есть ли у вас выезд мастера на дом и сколько он стоит?','unknown')
        self.assertLessEqual(set(re.findall(r'\[([^\]]+)\]',r['answer'])),{d['section'] for d in r['documents']})

    def test_missing_intake_section_is_explicitly_reported(self):
        a=Assistant()
        q='Расскажи, как передать ноутбук в ремонт: диагностика, приём, гарантия, выдача, проверка готовности и запись заранее.'
        r=a.ask(q,'regression',top_k=5)
        self.assertNotIn('intake', [d['section'] for d in r['documents']])
        text=r['answer'].lower()
        self.assertTrue(any(word in text for word in ['нет сведений','нет информации','не указан','не описан','отсутствует','отсутствуют','недостаточно']),r['answer'])
        self.assertIn('700',text)
        self.assertIn('согла',text)

if __name__=='__main__': unittest.main()
