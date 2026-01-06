def restock_rules(usage_level, avg_usage):
    if usage_level == "HIGH" and avg_usage > 5:
        return "RESTOCK IMMEDIATELY"
    elif usage_level == "HIGH":
        return "RESTOCK SOON"
    elif avg_usage > 3:
        return "MONITOR STOCK"
    else:
        return "STOCK OK"
