import pymysql
from Log import *


class DataBase:

    def __init__(self, db_info, log_obj=None):
        try:
            if log_obj is None:
                self.log = Logging()
                self.log.logConfig(account_id=db_info['db_username'])
            else:
                self.log = log_obj

            self.log.trace()

            self.db_host_name = db_info['db_host_name']
            self.db_username = db_info['db_username']
            self.db_user_password = db_info['db_user_password']
            self.db_name = db_info['db_name']
            self.db_port = db_info['db_port']
        except Exception as e:
            self.log.error('cant create database object: ', str(e))
            return

    def get_connection(self, ):
        try:
            if self.db_port is None:
                con = pymysql.connect(host=self.db_host_name, user=self.db_username,
                                      password=self.db_user_password, db=self.db_name)
            else:
                con = pymysql.connect(host=self.db_host_name, user=self.db_username, password=self.db_user_password,
                                      db=self.db_name, port=self.db_port)
            return con, None
        except Exception as e:
            self.log.error('cant create connection: ', str(e))
            return False, str(e)

    def select_query(self, query, args, mod=0):
        # mod=0 => return cursor
        # mod=1 => return cursor.fetchall()
        self.log.trace()
        if query == '':
            self.log.error('query in empty')
            return False
        con = None
        try:
            con, err = self.get_connection()
            if err is not None:
                raise Exception(err)
            db = con.cursor()
            db.execute(query, args)
            con.close()
        except Exception as e:
            self.log.error('except select_query', str(e))
            try:
                if con.open is True:
                    con.close()
            finally:
                return False

        if mod == 0:
            return db
        else:
            return db.fetchall()

    def select_query_dictionary(self, query, args, mod=0):
        # mod=0 => return cursor
        # mod=1 => return cursor.fetchall()
        self.log.trace()
        if query == '':
            self.log.error('query in empty')
            return False
        con = None
        try:
            con, err = self.get_connection()
            if err is not None:
                raise Exception(err)
            db = con.cursor(pymysql.cursors.DictCursor)
            db.execute(query, args)
            con.close()
        except Exception as e:
            self.log.error('except select_query_dictionary', str(e))
            try:
                if con.open is True:
                    con.close()
            finally:
                return False

        if mod == 0:
            return db
        else:
            return db.fetchall()

    def command_query(self, query, args, write_log=True):
        self.log.trace()
        if query == '':
            self.log.error('query in empty')
            return 'query is empty'
        con = None
        try:
            con, err = self.get_connection()
            if err is not None:
                raise Exception(err)

            db = con.cursor()
            db._defer_warnings = True
            db.autocommit = False
            db.execute(query, args)
            # db.executemany(query, args)
            con.commit()
            con.close()
            return True
        except Exception as e:
            if write_log is True:
                print('command_query. error:{0} query:{1}, args:{2}'.format(e, query, args))

                t = 'cant execute command_query_many and rollback. query:{0}'.format(query)
                if query.find('share_status') > 0:
                    t = 'cant execute command_query_many and rollback. query:{0}, args:{1}'.format(query, args)

                self.log.error(t, str(e))
                # self.log.error('cant execute command_query and rollback. query:{0}'.format(query), str(e))
            try:
                if con.open is True:
                    con.rollback()
                    con.close()
            finally:
                return 'cant execute command_query: {}'.format(str(e))

    def command_query_many(self, query, args, write_log=True):
        self.log.trace()
        if query == '':
            self.log.error('query in empty')
            return 'query in empty'
        con = None
        try:
            con, err = self.get_connection()
            if err is not None:
                raise Exception(err)

            db = con.cursor()
            db._defer_warnings = True
            db.autocommit = False
            # db.execute(query, args)
            db.executemany(query, args)
            con.commit()
            con.close()
            return True
        except Exception as e:
            if write_log is True:
                print('command_query_many. error:{0} query:{1}, args:{2}'.format(e, query, args))

                t = 'cant execute command_query_many and rollback. query:{0}'.format(query)
                if query.find('share_status') > 0:
                    t = 'cant execute command_query_many and rollback. query:{0}, args:{1}'.format(query, args)

                self.log.error(t, str(e))
            try:
                if con.open is True:
                    con.rollback()
                    con.close()
            finally:
                return 'cant execute command_query_many: {}'.format(str(e))

# ================================================================
    def get_symbol(self, coin_base=None):
        if coin_base is None:  # all symbol
            query = "select symbol, earlier_valid_timestamp, small_valid_interval from symbols"
            args = ()
        else:
            query = "select symbol, earlier_valid_timestamp, small_valid_interval from symbols where coin_base = %s"
            args = (coin_base)

        res = self.select_query(query=query, args=args, mod=1)

        return res

    def get_symbol_list(self, coin_base=None):
        res = self.get_symbol(coin_base=coin_base)
        if res is False:
            return []
        else:
            temp = []
            for symbol in res:
                # temp.append(symbol[0])
                temp.append(symbol)
            return temp

# ------------------------------------------------------
    def set_complete_candle_historical_data(self, interval, data):
        from binance.client import Client

        if interval == Client.KLINE_INTERVAL_1MINUTE:
            table_name = "symbol_historical_1MINUTE"
        elif interval == Client.KLINE_INTERVAL_3MINUTE:
            table_name = "symbol_historical_3MINUTE"
        elif interval == Client.KLINE_INTERVAL_5MINUTE:
            table_name = "symbol_historical_5MINUTE"
        elif interval == Client.KLINE_INTERVAL_15MINUTE:
            table_name = "symbol_historical_15MINUTE"
        elif interval == Client.KLINE_INTERVAL_30MINUTE:
            table_name = "symbol_historical_30MINUTE"
        elif interval == Client.KLINE_INTERVAL_1HOUR:
            table_name = "symbol_historical_1HOUR"
        elif interval == Client.KLINE_INTERVAL_2HOUR:
            table_name = "symbol_historical_2HOUR"
        elif interval == Client.KLINE_INTERVAL_4HOUR:
            table_name = "symbol_historical_4HOUR"
        elif interval == Client.KLINE_INTERVAL_6HOUR:
            table_name = "symbol_historical_6HOUR"
        elif interval == Client.KLINE_INTERVAL_8HOUR:
            table_name = "symbol_historical_8HOUR"
        elif interval == Client.KLINE_INTERVAL_12HOUR:
            table_name = "symbol_historical_12HOUR"
        elif interval == Client.KLINE_INTERVAL_1DAY:
            table_name = "symbol_historical_1DAY"
        elif interval == Client.KLINE_INTERVAL_3DAY:
            table_name = "symbol_historical_3DAY"
        elif interval == Client.KLINE_INTERVAL_1WEEK:
            table_name = "symbol_historical_1WEEK"
        elif interval == Client.KLINE_INTERVAL_1MONTH:
            table_name = "symbol_historical_1MONTH"
        else:
            err = "invalid interval"
            return err

        query = "insert IGNORE into {0} (symbol, open_time, open, high, low, close, volume, number_of_trade) " \
                "value (%s, %s, %s, %s, %s, %s, %s, %s)".format(table_name)
        args = data

        # print('args length:', len(args))
        if len(args) == 0:
            err = "empty data"
        else:
            err = self.command_query_many(query=query, args=args, write_log=True)
        return err

    def set_last_real_candle_historical_data(self, interval, data): #  نیاز به اصلاح نام جداول
        from binance.client import Client

        if interval == Client.KLINE_INTERVAL_1MINUTE:
            table_name = "symbol_historical_1MINUTE"
        elif interval == Client.KLINE_INTERVAL_3MINUTE:
            table_name = "symbol_historical_3MINUTE"
        elif interval == Client.KLINE_INTERVAL_5MINUTE:
            table_name = "symbol_historical_5MINUTE"
        elif interval == Client.KLINE_INTERVAL_15MINUTE:
            table_name = "symbol_historical_15MINUTE"
        elif interval == Client.KLINE_INTERVAL_30MINUTE:
            table_name = "symbol_historical_30MINUTE"
        elif interval == Client.KLINE_INTERVAL_1HOUR:
            table_name = "symbol_historical_1HOUR"
        elif interval == Client.KLINE_INTERVAL_2HOUR:
            table_name = "symbol_historical_2HOUR"
        elif interval == Client.KLINE_INTERVAL_4HOUR:
            table_name = "symbol_historical_4HOUR"
        elif interval == Client.KLINE_INTERVAL_6HOUR:
            table_name = "symbol_historical_6HOUR"
        elif interval == Client.KLINE_INTERVAL_8HOUR:
            table_name = "symbol_historical_8HOUR"
        elif interval == Client.KLINE_INTERVAL_12HOUR:
            table_name = "symbol_historical_12HOUR"
        elif interval == Client.KLINE_INTERVAL_1DAY:
            table_name = "symbol_historical_1DAY"
        elif interval == Client.KLINE_INTERVAL_3DAY:
            table_name = "symbol_historical_3DAY"
        elif interval == Client.KLINE_INTERVAL_1WEEK:
            table_name = "symbol_historical_1WEEK"
        elif interval == Client.KLINE_INTERVAL_1MONTH:
            table_name = "symbol_historical_1MONTH"
        else:
            err = "invalid interval"
            return err

        query = "insert IGNORE into {0} (symbol, open_time, open, high, low, close, volume, number_of_trade) " \
                "value (%s, %s, %s, %s, %s, %s, %s, %s)".format(table_name)
        args = data

        print('args length:', len(args))
        if len(args) == 0:
            err = "empty data"
        else:
            err = self.command_query_many(query=query, args=args, write_log=True)
        return err

    def get_complete_candle_historical_open_time(self, symbol, interval, start_datetime, end_datetime):
        from binance.client import Client

        if interval == Client.KLINE_INTERVAL_1MINUTE:
            table_name = "symbol_historical_1MINUTE"
        elif interval == Client.KLINE_INTERVAL_3MINUTE:
            table_name = "symbol_historical_3MINUTE"
        elif interval == Client.KLINE_INTERVAL_5MINUTE:
            table_name = "symbol_historical_5MINUTE"
        elif interval == Client.KLINE_INTERVAL_15MINUTE:
            table_name = "symbol_historical_15MINUTE"
        elif interval == Client.KLINE_INTERVAL_30MINUTE:
            table_name = "symbol_historical_30MINUTE"
        elif interval == Client.KLINE_INTERVAL_1HOUR:
            table_name = "symbol_historical_1HOUR"
        elif interval == Client.KLINE_INTERVAL_2HOUR:
            table_name = "symbol_historical_2HOUR"
        elif interval == Client.KLINE_INTERVAL_4HOUR:
            table_name = "symbol_historical_4HOUR"
        elif interval == Client.KLINE_INTERVAL_6HOUR:
            table_name = "symbol_historical_6HOUR"
        elif interval == Client.KLINE_INTERVAL_8HOUR:
            table_name = "symbol_historical_8HOUR"
        elif interval == Client.KLINE_INTERVAL_12HOUR:
            table_name = "symbol_historical_12HOUR"
        elif interval == Client.KLINE_INTERVAL_1DAY:
            table_name = "symbol_historical_1DAY"
        elif interval == Client.KLINE_INTERVAL_3DAY:
            table_name = "symbol_historical_3DAY"
        elif interval == Client.KLINE_INTERVAL_1WEEK:
            table_name = "symbol_historical_1WEEK"
        elif interval == Client.KLINE_INTERVAL_1MONTH:
            table_name = "symbol_historical_1MONTH"
        else:
            err = "invalid interval"
            return err

        query = "select open_time from {0} where symbol = %s and  open_time >= %s and open_time < %s".format(table_name)
        args = (symbol, start_datetime, end_datetime)
        # print(query)
        # print(args)
        return self.select_query(query=query, args=args, mod=1)

    def add_symbol(self, symbol, coin_base):
        query = "insert IGNORE into symbols (symbol, coin_base, earlier_valid_timestamp) value (%s, {0}, %s)".format(coin_base)
        args = symbol

        # print('args length:', len(args))
        if len(args) == 0:
            err = "empty data"
        else:
            err = self.command_query_many(query=query, args=args, write_log=True)
        return err

    def set_small_valid_interval(self, symbol, coin_base, small_valid_interval):
        query = 'update symbols set small_valid_interval = %s where symbol = %s and coin_base = %s'
        args = (small_valid_interval, symbol, coin_base)

        res = self.command_query(query=query, args=args, write_log=True)
        return res

if __name__ == "__main__":
    import app_setting
    import psutil

    db_server_id = 2

    print(psutil.disk_usage('D:'))
    # db = DataBase(db_info=app_setting.get_db_info(db_server_id=app_setting.client_id),log_obj=None)
    db = DataBase(db_info=app_setting.get_db_info(db_server_id=db_server_id),log_obj=None)

    # print(db.set_small_valid_interval(symbol='1INCHBTC', coin_base='BTC', small_valid_interval='1h'))
    print(db.get_symbol(coin_base='BTC'))
    print(db.get_symbol_list(coin_base='BTC'))


    # ---------------------
    # import api_setting
    # res = db.add_symbol(api_setting.symbol_ETH_list, "'ETH'")
    # print(res)
    # res = db.add_symbol(api_setting.symbol_BNB_list, "'BNB'")
    # print(res)
    # res = db.add_symbol(api_setting.symbol_BTC_list, "'BTC'")
    # print(res)
    # res = db.add_symbol(api_setting.symbol_USDT_list, "'USDT'")
    # print(res)
    # res = db.add_symbol(api_setting.symbol_TRX_list, "'TRX'")
    # print(res)
    # res = db.add_symbol(api_setting.symbol_XRP_list, "'XRP'")
    # print(res)

    # ---------------------
    # res = db.get_symbol(coin_base='BTC')
    # print(res)
    #
    # res = db.get_symbol_list(coin_base='BTC')
    # print(res)