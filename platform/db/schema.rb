# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 20180218024959) do

  create_table "anns", force: :cascade do |t|
    t.integer "flag", default: 1
    t.integer "votes", default: 1
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "articles", force: :cascade do |t|
    t.string "title"
    t.text "text"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "bboxes", force: :cascade do |t|
    t.float "x0"
    t.float "y0"
    t.float "x1"
    t.float "y1"
    t.string "word"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "images", force: :cascade do |t|
    t.string "text_image_file_name"
    t.string "text_image_content_type"
    t.integer "text_image_file_size"
    t.datetime "text_image_updated_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

end
