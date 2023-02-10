import xlsxwriter


workbook = xlsxwriter.Workbook('exported.xlsx')
worksheet = workbook.add_worksheet()

row, column = 0, 1
with open('measure_result_2.txt') as reader:
    while True:
        line = reader.readline()
        if not line:
            break

        if line.startswith('PSNR:'):
            parts = line.split(' ')
            value = round(float(parts[1]), 3)
            worksheet.write(row, column, value)
            line = reader.readline()
            if line.startswith('SSIM:'):
                parts = line.split(' ')
                value = round(float(parts[1]), 3)
                worksheet.write(row, column - 1, value)

                row += 1

            if row == 5:
                row += 1
            if row == 10:
                row += 1
            if row == 19:
                row += 2
            if row == 33:
                row += 1
            if row == 37:
                row += 1
            if row == 50:
                row += 1
            if row == 75:
                column += 2
                row = 0

workbook.close()
