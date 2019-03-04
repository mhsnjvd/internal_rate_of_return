import numpy as np
import pandas as pd
import numbers
from datetime import date, timedelta, datetime
from scipy.optimize import root_scalar


COLUMN_NAME_MAP = {
    'ticker':              'Company Symbol',
    'Name':                'Name',
    'Sector':              'Sector',
    'BVPS':                'BVPS',
    'EPS0':                'EPS0',
    'EPS1':                'EPS1',
    'EPS2':                'EPS2',
    'EPS3':                'EPS3',
    'EPS4':                'EPS4',
    'EPS5':                'EPS5',
    'DPS0':                'DPS0',
    'DPS1':                'DPS1',
    'DPS2':                'DPS2',
    'DPS3':                'DPS3',
    'DPS4':                'DPS4',
    'DPS5':                'DPS5',
    'Previous year end':   'Previous year end',
    'Year end':            'Year end',
    'Price':               'Price',
}


class Data():
    """Simple wrapper class for maintaining a single copy
    of the data"""

    def __init__(self, file_name ):
        pass

    @staticmethod
    def convert_int64_to_date(date_int):
        """
        :param date_int: an integer for the form 20180121
        :return: a date
        """
        date = datetime.strptime(str(date_int), '%Y%m%d').date()
        return date

    @staticmethod
    def convert_string_to_double(number_str):
        """
        :param number_str: a string of the format 123,456.00
        :return: a floating point number 123456.00
        """
        return pd.to_numeric(number_str.replace(',', ''))

    @staticmethod
    def get_data_frame():
        file_name = 'data.csv'
        str = ' '.join(['Reading file', file_name])
        print(str)
        df = pd.read_csv(file_name)
        print( 'Done reading ')
        df[COLUMN_NAME_MAP['Year end']] = df[COLUMN_NAME_MAP['Year end']].apply( Data.convert_int64_to_date )
        df[COLUMN_NAME_MAP['Previous year end']] = df[COLUMN_NAME_MAP['Previous year end']].apply( Data.convert_int64_to_date )
        df[COLUMN_NAME_MAP['Price']] = df[COLUMN_NAME_MAP['Price']].apply(Data.convert_string_to_double)

        return df

class Company():
    def get_current_date(self):
        """
        Give me the current date
        :return: a datetime object for the current date
        """
        td = timedelta(days=0)
        return date.today() + td
    def set_current_date(self, input_date=date.today()):
        """
        Setter for current date
        :param date: set date to the input date
        :return: nothing
        """
        if not isinstance(input_date, date):
            raise ValueError("input date must be of type datetime.date")
        td = timedelta(days=0)
        self.current_date = input_date + td

    def set_lt_growth(self, value=.04):
        """
        setter for lt_growth
        :param value:
        :return:
        """
        if not isinstance(value, numbers.Number):
            raise ValueError("LT growth must be a number")
        self.lt_growth = value

    def get_lt_growth(self):
        return self.lt_growth

    def get_previous_year_end_date(self):
        return self.previous_year_end_date

    def get_bvps(self):
        return self.bvps

    def get_price(self):
        return self.price

    def get_price_premium(self):
        bvps_now = self.get_bvps_now()
        price = self.get_price()
        return bvps_now - price


    def get_next_year_end_date(self):
        previous_year_end_date = self.get_previous_year_end_date()
        return datetime(year=previous_year_end_date.year + 1,
                        month=previous_year_end_date.month,
                        day=previous_year_end_date.day).date()

    def get_eps1_adjusted(self, eps1, eps2, number_of_days):
        if number_of_days >= 0:
            return eps2
        else:
            return eps1

    def get_froe(self, eps1_adjusted, bvps):
        """
        FROE := eps1_adjusted/bvps
        :return:  FROE
        """
        froe = eps1_adjusted / bvps
        return froe

    def get_bvps_now(self, bvps, froe, number_of_days):
        """
        Get the book value per share now
        The formula used is:
        BVPS_NOW = BVPS * (1 + FROE) ^ ( days since last reporting / 365 )
        :return: a double BVPS_NOW
        """
        day_count_fraction = number_of_days / 365.0
        #TODO Some bvps are negative and 1 + froe is also negative
        bvps_now = bvps * ( np.abs(1 + froe) ** day_count_fraction )
        assert not np.isnan(bvps_now), 'Calculation resulted in nan'
        return bvps_now

    def get_eps_between_now_and_next_year_end(self, eps1_adjusted, bvps_now, bvps):
        val = eps1_adjusted - (bvps_now - bvps)
        return val

    def get_value(self, property, data_row):
        """
        :param property: the property that we want to querry, a string
        :param data_row: a dataframe with one row only
        :return: returns the value of the dataframe column specified by 'property'
        """
        assert len(data_row) == 1, 'data row must be of length 1'
        val = data_row[COLUMN_NAME_MAP[property]].values[0]
        return val

    def get_days_elapsed_since_reporting_year_end(self):
        current_date = self.get_current_date()
        reporting_year_end_date = self.previous_year_end_date
        assert current_date >= reporting_year_end_date, 'current date must not be less than the reporting year end'
        time_delta = current_date - reporting_year_end_date
        return time_delta.days

    def get_days_remaining_till_year_end(self):
        current_date = self.get_current_date()
        next_year_end_date = self.get_next_year_end_date()
        # current_date can actually be greater than the next_year_end_date,
        # you can see the example of KMX
        # assert current_date <= next_year_end_date, 'current date must not be greater than the next year end'
        time_delta = next_year_end_date - current_date
        return time_delta.days

    def get_date_from_string(self, date_str):
        return datetime.strptime(date_str, '%Y%m%d').date()

    @staticmethod
    def nan_to_zero(x):
        """
        Static method which returns a zero if the input x is a nan,
        returns x otherwise
        :param x: a float
        :return: a float
        """
        if np.isnan(x):
            return 0.0
        else:
            return x

    @staticmethod
    def zero_to_nan(x):
        """
        if input x is a zero, return a numpy nan, return x otherwise
        :param x: a float
        :return: a numpy nan or a float
        """
        if x == 0.0:
            return np.nan
        else:
            return x

    def get_clean_company_data(self):
        eps_pass_1 = np.zeros(5)
        eps_pass_1[0] = Company.zero_to_nan(self.eps1)
        eps_pass_1[1] = Company.zero_to_nan(self.eps2)
        eps_pass_1[2] = Company.zero_to_nan(self.eps3)
        eps_pass_1[3] = Company.zero_to_nan(self.eps4)
        eps_pass_1[4] = Company.zero_to_nan(self.eps5)

        dps_pass_1 = np.zeros(5)
        dps_pass_1[0] = self.dps1
        dps_pass_1[1] = self.dps2
        dps_pass_1[2] = self.dps3
        dps_pass_1[3] = self.dps4
        dps_pass_1[4] = self.dps5

        eps_pass_2 = np.zeros(5)

        number_of_days = self.get_days_elapsed_since_reporting_year_end()
        eps1_adj = self.get_eps1_adjusted(eps_pass_1[0], eps_pass_1[1], number_of_days)

        if eps1_adj == eps_pass_1[1]:
            eps_pass_2[0] = eps_pass_1[2]
        else:
            eps_pass_2[0] = eps_pass_1[1]

        if eps_pass_2[0] == eps_pass_1[2]:
            eps_pass_2[1] = eps_pass_1[3]
        else:
            eps_pass_2[1] = eps_pass_1[2]

        if eps_pass_2[1] == eps_pass_1[3]:
            eps_pass_2[2] = eps_pass_1[4]
        else:
            eps_pass_2[2] = eps_pass_1[3]

        if eps_pass_2[2] == eps_pass_1[4]:
            eps_pass_2[3] = np.nan
        else:
            eps_pass_2[3] = eps_pass_1[4]

        # This is left empty in the excel sheet: cell C55
        # We are forcing it to a nan
        eps_pass_2[4] = np.nan

        bvps = self.get_bvps()
        froe = self.get_froe(eps1_adj, bvps)
        bvps_now = self.get_bvps_now(bvps, froe, number_of_days)
        eps_between_now_and_next_year = self.get_eps_between_now_and_next_year_end(eps1_adj, bvps_now, bvps)

        dps_pass_2 = np.zeros(5)

        if self.days_elapsed_since_reporting_year_end >= 0:
            dps_pass_2[0] = dps_pass_1[1]
        else:
            dps_pass_2[0] = dps_pass_1[0]

        if dps_pass_2[0] == dps_pass_1[1]:
            dps_pass_2[1] = dps_pass_1[2]
        else:
            dps_pass_2[1] = dps_pass_1[1]

        if dps_pass_2[1] == dps_pass_1[2]:
            dps_pass_2[2] = dps_pass_1[3]
        else:
            dps_pass_2[2] = dps_pass_1[2]

        if dps_pass_2[2] == dps_pass_1[3]:
            dps_pass_2[3] = dps_pass_1[4]
        else:
            dps_pass_2[3] = dps_pass_1[3]

        if dps_pass_2[3] == dps_pass_1[4]:
            dps_pass_2[4] = np.nan
        else:
            dps_pass_2[4] = dps_pass_1[4]

        cleaned_data = {
            'eps_pass_1':                       eps_pass_1,
            'eps_pass_2':                       eps_pass_2,
            'dps_pass_1':                       dps_pass_1,
            'dps_pass_2':                       dps_pass_2,
            'eps1_adj':                         eps1_adj,
            'eps_between_now_and_next_year':    eps_between_now_and_next_year,
            'froe':                             froe,
            'bvps_now':                         bvps_now,
        }

        return cleaned_data


    def get_roll_forward_data(self):
        cleaned_data = self.get_clean_data()
        eps_pass_1 = cleaned_data['eps_pass_1']
        eps_pass_2 = cleaned_data['eps_pass_2']


        eps_forecasts = np.zeros(5)
        eps_forecasts[0] = cleaned_data['eps_between_now_and_next_year']
        eps_forecasts[1] = eps_pass_1[1]

        if np.isnan(eps_pass_1[2]):
            eps_forecasts[2] = eps_forecasts[1] + (eps_forecasts[1] - eps_forecasts[0]) / 2.0
        else:
            eps_forecasts[2] = eps_pass_1[2]

        if np.isnan(eps_pass_2[2]):
            eps_forecasts[3] = eps_forecasts[2] + (eps_forecasts[2] - eps_forecasts[0]) / 3.0
        else:
            eps_forecasts[3] = eps_pass_2[2]

        if np.isnan(eps_pass_2[3]):
            eps_forecasts[4] = eps_forecasts[3] + (eps_forecasts[3] - eps_forecasts[0]) / 4.0
        else:
            eps_forecasts[4] = eps_pass_2[3]

        dps_pass_1 = cleaned_data['dps_pass_1']
        dps_pass_2 = cleaned_data['dps_pass_2']
        dps_forecasts = np.zeros(5)
        dps_forecasts[0] = dps_pass_2[0]
        dps_forecasts[1] = dps_pass_2[1]

        if np.isnan(dps_pass_2[2]):
            dps_forecasts[2] = dps_forecasts[1] + (dps_forecasts[1] - dps_forecasts[0]) / 2.0
        else:
            dps_forecasts[2] = dps_pass_2[2]

        if np.isnan(dps_pass_2[3]):
            dps_forecasts[3] = dps_forecasts[2] + (dps_forecasts[2] - dps_forecasts[0]) / 3.0
        else:
            dps_forecasts[3] = dps_pass_2[3]

        if np.isnan(dps_pass_2[4]):
            dps_forecasts[4] = dps_forecasts[3] + (dps_forecasts[3] - dps_forecasts[0]) / 4.0
        else:
            dps_forecasts[4] = dps_pass_2[4]

        bvps_list = np.zeros(6)
        bvps = self.get_bvps()
        eps1_adj = cleaned_data['eps1_adj']
        bvps_now = cleaned_data['bvps_now']
        bvps_list[0] = bvps_now

        dps1 = dps_pass_2[0]
        bvps_list[1] = bvps + eps1_adj - dps1

        for i in range(2, len(bvps_list)):
            bvps_list[i] =  bvps_list[i-1] + eps_forecasts[i-1] - dps_forecasts[i-1]

        return {
            'opening_book_values': bvps_list,
            'eps_forecasts':       eps_forecasts,
            'dps_forecasts':       dps_forecasts,
            'bvps':                bvps,
            'eps1_adj':            eps1_adj,
        }

    def get_growth_in_book_values(self, opening_book_values, lt_growth, n_iterations=30):
        n = len(opening_book_values) - 1
        growth_in_book_values = np.zeros(n_iterations + n)
        growth_in_book_values[:n] = np.diff(opening_book_values) / opening_book_values[:-1]
        gbv = growth_in_book_values[n-1]

        ratio = lt_growth / gbv
        for i in range(n, len(growth_in_book_values)):
            growth_in_book_values[i] = (ratio ** (1.0/n_iterations)) * growth_in_book_values[i-1]

        return growth_in_book_values

    def get_book_values_from_growth_in_book_values(self, growth_in_book_values, opening_book_values):
        n = len(opening_book_values) - 1
        book_values = np.zeros(growth_in_book_values.shape)
        book_values[:n+1] = opening_book_values
        for i in range(n+1,len(book_values)):
            book_values[i] = book_values[i-1] * (1 + growth_in_book_values[i-1])

        return book_values

    def get_roes(self, irr, bvps, eps1_adj, eps_forecasts, opening_book_values, n_iterations=30):
        n = len(eps_forecasts)
        roes = np.zeros(n_iterations + n)
        roes[0] = eps1_adj / bvps
        roes[1:n] = eps_forecasts[1:n] / opening_book_values[1:n]
        ratio = irr / roes[n-1]
        for i in range(n, len(roes)):
            roes[i] = (ratio ** (1.0/n_iterations)) * roes[i-1]

        return roes

    def get_residual_incomes(self, book_values, roes, irr, eps_forecasts, n_days):
        """
        Get residual incomes
        :param book_values: numpy array of length n
        :param roes:  numpy array of length n
        :param irr:  a floating point number
        :param eps_forecasts: a numpy array, only first value is used
        :param n_days: an integer
        :return: a numpy array of length n
        """
        assert len(book_values) == len(roes), "length of books values must be the same as the length of roes"
        residual_incomes = book_values * (roes - irr)
        #Adjust the first value:
        residual_incomes[0] = eps_forecasts[0] - ((1 + irr) ** ((365 - n_days) / 365) - 1) * book_values[0]

        return residual_incomes

    def get_pvs_of_residual_incomes(self, residual_incomes, irr, n_days):
        """
        returns a numpy array with the present values of residual incomes
        :param residual_incomes: a numpy array of length n
        :param irr: internal rate of retrun, a double
        :param n_days: an integer
        :return: numpy array of length n
        """
        periods = np.arange(0,len(residual_incomes))
        factors = 1.0/((1+irr) ** (periods + (365-n_days)/365))
        pvs = factors * residual_incomes
        return pvs

    def get_npv(self, irr, book_values, number_of_days, eps1_adj, price, bvps, eps_forecasts, opening_book_values):
        roes = self.get_roes(irr, bvps, eps1_adj, eps_forecasts, opening_book_values)
        residual_incomes = self.get_residual_incomes( book_values, roes, irr, eps_forecasts, number_of_days)
        pv_residual_incomes = self.get_pvs_of_residual_incomes(residual_incomes, irr, number_of_days)
        npv = np.sum(pv_residual_incomes) + opening_book_values[0] - price
        return npv


    def solve_irr(self, number_of_days, eps1_adj, price, bvps, eps_forecasts, dps_forecasts, opening_book_values, lt_growth):
        growth_in_book_values = self.get_growth_in_book_values(opening_book_values, lt_growth)
        book_values = self.get_book_values_from_growth_in_book_values(growth_in_book_values, opening_book_values)

        # Define a scalar function that will be used by the root solver
        f = lambda irr: self.get_npv(irr, book_values, number_of_days, eps1_adj, price, bvps, eps_forecasts, opening_book_values)

        # Call the root solver, make sure the left end point of the bracket is greater than zero, otherwise
        # the iterations can fail as f can not be evaluated with zero input
        internal_rate_of_return = root_scalar(f, bracket=[1e-4, 2.0])
        return internal_rate_of_return

    def get_residual_frame(self):
        roll_forward_data = self.get_roll_forward_data()
        opening_book_values = roll_forward_data['opening_book_values']
        eps_forecasts = roll_forward_data['eps_forecasts']
        dps_forecasts = roll_forward_data['dps_forecasts']
        bvps = roll_forward_data['bvps']
        eps1_adj = roll_forward_data['eps1_adj']
        price = self.get_price()
        lt_growth = self.get_lt_growth()
        gbvs = self.get_growth_in_book_values( opening_book_values, lt_growth)
        book_values = self.get_book_values_from_growth_in_book_values(gbvs, opening_book_values)
        roes = self.get_roes(irr, bvps, eps1_adj, eps_forecasts,  opening_book_values)
        number_of_days = 305
        residual_incomes = self.get_residual_incomes( book_values, roes, irr, eps_forecasts, number_of_days)
        pv_residual_incomes = self.get_pvs_of_residual_incomes(residual_incomes, irr, number_of_days)
        npv = self.get_npv( irr, book_values, number_of_days, eps1_adj, price, bvps, eps_forecasts, opening_book_values)
        r = self.solve_irr( number_of_days, eps1_adj, price, bvps, eps_forecasts, dps_forecasts, opening_book_values, lt_growth)
        return r

    def get_clean_data(self):
        return self.cleaned_data

    def __init__(self, data_row, lt_growth=0.04, current_date=date.today()):
        """
        Initialize the company object from a dataframe which
        has only one row
        :param data_row:
        :return nothing, just initialize and clean the data
        """
        assert len(data_row) == 1, 'data of a company must be in a single row'

        # Name data for the company
        self.name   = self.get_value('Name',   data_row)
        self.ticker = self.get_value('ticker', data_row)

        # Set current date for the company
        self.set_current_date(current_date)

        # Set the long term growth rate
        self.set_lt_growth(lt_growth)

        # Earnings per share data
        self.eps0 = self.get_value('EPS0', data_row)
        self.eps1 = self.get_value('EPS1', data_row)
        self.eps2 = self.get_value('EPS2', data_row)
        self.eps3 = self.get_value('EPS3', data_row)
        self.eps4 = self.get_value('EPS4', data_row)
        self.eps5 = self.get_value('EPS5', data_row)

        # Dividends per share data
        self.dps0 = self.get_value('DPS0', data_row)
        self.dps1 = self.get_value('DPS1', data_row)
        self.dps2 = self.get_value('DPS2', data_row)
        self.dps3 = self.get_value('DPS3', data_row)
        self.dps4 = self.get_value('DPS4', data_row)
        self.dps5 = self.get_value('DPS5', data_row)

        # Price and book value per share
        self.price = self.get_value('Price', data_row)
        self.bvps  = self.get_value('BVPS', data_row)

        # useful dates
        # [TODO] Confusion here, what exactly is previous year?
        self.previous_year_end_date = self.get_value('Year end', data_row)
        self.next_year_end_date     = self.get_next_year_end_date()
        self.days_elapsed_since_reporting_year_end = self.get_days_elapsed_since_reporting_year_end()
        self.days_remaining_till_year_end = self.get_days_remaining_till_year_end()
        self.cleaned_data = self.get_clean_company_data()


    def date_to_str(self, dt):
        """
        Convert date to string of the format %Y-%m-%d
        :param dt: date object
        :return: string
        """
        return datetime.strftime( dt, '%Y-%m-%d')

    def __repr__(self):
        return ' '.join([
            'Company:',        self.name,
            ', Ticker:',       self.ticker,
            ', Current Date:', self.date_to_str(self.get_current_date()),
            ', LT Growth:',    str(self.get_lt_growth()),
        ])

    def __str__(self):
        return self.__repr__()

class IRR():
    data = Data.get_data_frame()
    def __init__(self):
        self.tickers = self.get_all_company_tickers()
        self.company_objects = [ self.get_company_object(t) for t in self.tickers ]

    def get_company_object( self, ticker):
        df = IRR.data
        company_data_row = df[ df[COLUMN_NAME_MAP['ticker']] == ticker ]
        if len(company_data_row) != 1:
            raise Exception('size of data is more than one')

        return Company(company_data_row)

    def get_all_company_tickers(self):
        df = IRR.data
        return df['Company Symbol'].unique()

    def get_residual_income_analysis(self, ticker, current_date, company_data, lt_growth):
        company = Company(company_data, current_date, lt_growth)
        company.solve_irr()

    def analyse_company(self, ticker):
        company_data = df[ df[COLUMN_NAME_MAP['ticker']] == ticker ]
        company_df = self.get_residual_income_analysis(ticker, date.today(), company_data, 0.04)


def main():
    pass

if __name__ == "__main__":
    irr  = IRR()
    irr2 = IRR()
    all_tickers = irr.get_all_company_tickers()
    all_companies = [irr.get_company_object(x) for x in all_tickers]
    all_companies[0]
    [print(x) for x in all_companies]
    df = irr.data.copy(deep=True)
    c2 = irr.get_company_object('FB')
    c = irr.get_company_object('NWSA')
    l = irr.company_objects

    opening_book_values = np.array([30.44, 37.96, 46.78, 57.27, 70.00, 84.03])
    eps_forecasts = np.array([7.52, 8.82, 10.50, 12.73, 14.03])
    irr = 9.34151177372357/100.0
    bvps = 29.15
    eps1_adj = 8.82
    price = 161.5
    dps_forecasts = 0*eps_forecasts
    lt_growth = .04

    good = 0
    bad  = 0
    for x in l:
        try:
            x.get_residual_frame()
            good = good + 1
        except:
            print(x.name)
            bad = bad + 1


    print(good)
    print(bad)





#     gbvs = get_growth_in_book_values( opening_book_values, lt_growth)
#     book_values = get_book_values_from_growth_in_book_values(gbvs, opening_book_values)
#     roes = get_roes(irr, bvps, eps1_adj, eps_forecasts,  opening_book_values)
#     number_of_days = 305
#     residual_incomes = get_residual_incomes( book_values, roes, irr, eps_forecasts, number_of_days)
#     pv_residual_incomes = get_pvs_of_residual_incomes(residual_incomes, irr, number_of_days)
#     npv = get_npv( irr, book_values, number_of_days, eps1_adj, price, bvps, eps_forecasts, opening_book_values)
#     r = solve_irr( number_of_days, eps1_adj, price, bvps, eps_forecasts, dps_forecasts, opening_book_values, lt_growth)
#     print(r)
#     rfd = c2.get_roll_forward_data()
#     print(rfd)


