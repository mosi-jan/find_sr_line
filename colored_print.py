from termcolor import colored

class colored_print():
    def __init__(self, default_text_color=None):
        self.default_text_color = default_text_color

        # if default_text_color is None:
        #     self.default_text_color = None  # white
        # else:
        #     self.default_text_color=default_text_color

    def print(self, text, color=None):
        try:
            if  color is None:
                print(colored(text, self.default_text_color))
            else:
                print(colored('| ', self.default_text_color) + colored(text, color))

        except Exception as e:
            self.print('invalid color: %s'% str(e), 'red')


if __name__ == "__main__":
    p = colored_print()
    p.default_text_color = 'red'

    # diff_print_color = 'red'
    print(1)
    # p.print_c('2')
    p.print('p', 'blue')
    # p.print_c('2', 'blue')