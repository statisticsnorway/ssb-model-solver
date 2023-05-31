class Equation:
    _lag_notation = '__LAG'

    def __init__(self, eqn_str):
        self._eqn_str = eqn_str.lower()
        self.parsed_eqn, self.var_mapping, self.lag_mapping = self._analyze_eqn_str(eqn_str.lower())

    def _analyze_eqn_str(self, eqn: str):
        parsed_eqn_with_lag_notation, var_mapping, lag_mapping = [], {}, {}
        component, lag = '', ''
        is_num, is_var, is_lag, is_sci = False, False, False, False

        for chr in ''.join([eqn, ' ']):
            is_num = (chr.isnumeric() and not is_var) or is_num
            is_var = (chr.isalpha()  and not is_num) or is_var
            is_lag = (is_var and chr == '(') or is_lag
            is_sci = (is_num and chr == 'e') or is_sci

            if (is_var and chr == '(' and component in ['max', 'min', 'log', 'exp']):
                parsed_eqn_with_lag_notation += ''.join([component, chr])
                is_var, is_lag = False, False
                component, lag = '', ''
                continue

            # Check if character is something other than a numeric, variable or lag and write numeric or variable to parsed equation
            if chr in ['=','+','-','*','/','(',')',',',' '] and not (is_lag or is_sci):
                if is_num:
                    parsed_eqn_with_lag_notation += str(component),
                if is_var:
                    # Replace (-)-notation by LAG_NOTATION for lags and appends _ to the end to mark the end
                    pfx = '' if lag == '' else ''.join([Equation._lag_notation, str(-int(lag[1:-1])), '_'])
                    parsed_eqn_with_lag_notation += ''.join([component, pfx]),
                    var_mapping[''.join([component, lag])] = ''.join([component, pfx])
                    var_mapping[''.join([component, pfx])] = ''.join([component, lag])
                    lag_mapping[''.join([component, pfx])] = (component, 0 if lag == '' else -int(lag[1:-1]))
                if chr != ' ':
                    parsed_eqn_with_lag_notation += chr,
                component, lag = '', ''
                is_num, is_var, is_lag = False, False, False
                continue

            if is_sci and chr.isnumeric():
                is_sci = False

            if is_num:
                component = ''.join([component, chr])
                continue

            if is_var and not is_lag:
                component = ''.join([component, chr])
                continue

            if is_var and is_lag:
                lag = ''.join([lag, chr])
                if chr == ')':
                    is_lag = False

        return parsed_eqn_with_lag_notation, var_mapping, lag_mapping