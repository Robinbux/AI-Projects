from Flags.Flags import Flags


def print_response(response):
    Flags.verbose_print("--------------")
    Flags.verbose_print("   RESPONSE   ")
    Flags.verbose_print("--------------")
    Flags.verbose_print(response.to_pandas_dataframe().head(10))
    Flags.verbose_print("")
