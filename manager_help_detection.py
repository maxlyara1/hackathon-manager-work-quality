class ManagerHelped:
    def __init__(self):
        pass

    def analyze_manager_text(self, client_text):
        result = any(word in client_text.lower() for word in ['спасибо', 'хорошо', 'помогли', 'выручили'])
        return result
