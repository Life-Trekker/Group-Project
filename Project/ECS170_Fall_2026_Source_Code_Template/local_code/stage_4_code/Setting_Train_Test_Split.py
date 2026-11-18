'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting

class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        # load dataset (returns {'train': {...}, 'test': {...}})
        loaded_data = self.dataset.load()

        # pass both splits through untouched, so extra keys such as 'lengths' survive
        self.method.data = {
            'train': dict(loaded_data['train']),
            'test': dict(loaded_data['test']),
        }
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate_all()