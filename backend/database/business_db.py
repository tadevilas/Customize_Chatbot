import sqlite3
import os

BUSINESS_DB_PATH = os.path.join("data", "business_data.db")

def populate_database(db_path: str = BUSINESS_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Products Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL
        )
    """)

    # Customers Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            region TEXT NOT NULL,
            join_date TEXT NOT NULL
        )
    """)

    # Sales Orders Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales_orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)

    # Clear and insert sample records
    cursor.execute("DELETE FROM sales_orders")
    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM customers")

    cursor.executemany("""
        INSERT INTO products (product_id, name, category, price, stock_quantity) VALUES (?, ?, ?, ?, ?)
    """, [
        (1, "AI Developer Laptop Pro", "Electronics", 1899.99, 45),
        (2, "Cloud GPU Workstation", "Hardware", 4299.00, 12),
        (3, "4K Ultra Monitor 32inch", "Electronics", 499.50, 80),
        (4, "Mechanical Ergonomic Keyboard", "Accessories", 129.99, 150),
        (5, "Noise-Canceling AI Headset", "Accessories", 199.99, 95),
        (6, "Enterprise LLM Gateway Server", "Servers", 8900.00, 6)
    ])

    cursor.executemany("""
        INSERT INTO customers (customer_id, name, email, region, join_date) VALUES (?, ?, ?, ?, ?)
    """, [
        (1, "Alice Smith", "alice@enterprise-ai.com", "North America", "2023-01-15"),
        (2, "Bob Jones", "bob@fintech-solutions.org", "Europe", "2023-04-20"),
        (3, "Carlos Santana", "carlos@globaltech.co", "Latin America", "2023-07-10"),
        (4, "Diana Prince", "diana@cloudscale.io", "Asia-Pacific", "2023-11-05"),
        (5, "Evan Wright", "evan@ai-research-lab.net", "North America", "2024-02-18")
    ])

    cursor.executemany("""
        INSERT INTO sales_orders (order_id, customer_id, product_id, quantity, total_amount, order_date) VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (101, 1, 1, 2, 3799.98, "2024-03-01"),
        (102, 1, 3, 2, 999.00, "2024-03-01"),
        (103, 2, 2, 1, 4299.00, "2024-03-15"),
        (104, 3, 4, 5, 649.95, "2024-04-02"),
        (105, 4, 5, 3, 599.97, "2024-04-10"),
        (106, 5, 6, 1, 8900.00, "2024-04-18"),
        (107, 2, 1, 1, 1899.99, "2024-05-05")
    ])

    conn.commit()
    conn.close()

def init_business_database():
    populate_database(BUSINESS_DB_PATH)

if __name__ == "__main__":
    init_business_database()
    print(f"Business database initialized successfully at {BUSINESS_DB_PATH}!")
