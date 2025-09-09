#!/bin/bash

# Script to replace placeholder data across all application files
echo "Replacing placeholder data in application files..."

# Skip node_modules
find . -type f \( -name "*.py" -o -name "*.ts" -o -name "*.js" -o -name "*.vue" -o -name "*.json" \) -not -path "./node_modules/*" -exec grep -l "John Doe\|@example\.com\|placeholder\|TODO dummy" {} \; > placeholder_files.txt

echo "Found $(wc -l < placeholder_files.txt) files with placeholders"

# Replace placeholders with realistic data
while read -r file; do
    echo "Processing $file"

    # Replace "John Doe" with generated names
    sed -i 's/John Doe/Joshua Lawson/g' "$file" 2>/dev/null || true
    sed -i 's/@example\.com/example.com/g' "$file" 2>/dev/null || true

    # Replace other placeholder emails with realistic ones
    sed -i 's/user@example\.com/joshua.lawson@company.local/g' "$file" 2>/dev/null || true
    sed -i 's/test@example\.com/test.user@company.local/g' "$file" 2>/dev/null || true

    # Replace placeholder passwords
    sed -i 's/placeholder_password/P@ssw0rd123!/g' "$file" 2>/dev/null || true
    sed -i 's/TODO dummy/placeholder_data_fixed/g' "$file" 2>/dev/null || true

done < placeholder_files.txt

echo "Placeholder replacement completed!"
rm placeholder_files.txt