from gui import WindowApp
import os

if __name__ == '__main__':
    os.makedirs('debugs', exist_ok=True)
    models = [
        ['辨識數字的模型', 'digit.pt'],
        ['辨識英文的模型', 'lowerLetter.pt'],
        ['辨識數字和英文的混合模型', 'mix_digit-lowerLetter.pt'],
    ]
    WindowApp(models).start()
